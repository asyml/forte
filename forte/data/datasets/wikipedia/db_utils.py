# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A set of utilities to support reading DBpedia datasets.
"""
import bz2
import logging
import os
import re
import sys
from collections import OrderedDict
from random import choice
from typing import List, Dict, Tuple, Union, Any, Iterator
from urllib.parse import urlparse, parse_qs

import rdflib

dbpedia_prefix = "http://dbpedia.org/resource/"
state_type = Tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]


def load_redirects(redirect_path: str) -> Dict[str, str]:
    redirect_to = {}
    redirect_rel = "http://dbpedia.org/ontology/wikiPageRedirects"

    count = 0
    for statements in NIFParser(redirect_path):
        for statement in statements:
            s, v, o, _ = statement
            if str(v) == redirect_rel:
                count += 1
                from_page = get_resource_name(s)
                redirect_page = get_resource_name(o)
                redirect_to[from_page] = redirect_page
                if count % 50000 == 0:
                    logging.info("Loaded %d redirects.", count)
    return redirect_to


def get_resource_attribute(url, param_name) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query)[param_name][0]


def context_base(c: rdflib.Graph) -> str:
    return strip_url_params(c.identifier)


def get_resource_fragment(url) -> str:
    return urlparse(url).fragment


def get_resource_name(url) -> str:
    parsed = urlparse(url)
    return re.sub('^/resource/', '', parsed.path)


def strip_url_params(url) -> str:
    parsed = urlparse(url)
    # scheme + netloc + path
    return parsed.scheme + "://" + parsed.netloc + parsed.path


def print_progress(msg: str, end='\r'):
    """
    Print progress message to the same line.
    Args:
        msg: The message to print
        end: Line ending in terminal

    Returns:

    """
    sys.stdout.write("\033[K")  # Clear to the end of line.
    print(f' -- {msg}', end=end)


def print_notice(msg: str):
    """
    Print additional notice in a new line.
    Args:
        msg: The message to print

    Returns:

    """
    print(f'\n -- {msg}')


class NIFParser:
    def __init__(self, nif_path, tuple_format='nquads'):
        if nif_path.endswith(".bz2"):
            self.__nif = bz2.BZ2File(nif_path)
        else:
            self.__nif = open(nif_path)

        self.format = tuple_format

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def parse_graph(self, data: str, tuple_format: str) -> List:
        if self.format == 'nquads':
            g_ = rdflib.ConjunctiveGraph()
        else:
            g_ = rdflib.Graph()

        g_.parse(data=data, format=tuple_format)

        if self.format == 'nquads':
            return list(g_.quads())
        else:
            return list(g_)

    def read(self):
        while True:
            line = next(self.__nif)
            statements = list(self.parse_graph(
                line.decode('utf-8'),
                tuple_format=self.format
            ))

            if len(statements) > 0:
                return list(statements)

    def close(self):
        self.__nif.close()


class ContextGroupedNIFReader:
    def __init__(self, nif_path: str):
        self.__parser = NIFParser(nif_path)
        self.data_name = os.path.basename(nif_path)

        self.__last_c: str = ''
        self.__statements: List[state_type] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__parser.close()

    def __iter__(self):
        return self

    def __next__(self):
        res_c: str = ''
        res_states: List = []

        while True:
            try:
                for statements in self.__parser:
                    for s, v, o, c in statements:
                        c_ = context_base(c)

                        if c_ != self.__last_c and self.__last_c != '':
                            res_c = self.__last_c
                            res_states.extend(self.__statements)
                            self.__statements.clear()

                        self.__statements.append((s, v, o))
                        self.__last_c = c_

                        if not res_c == '':
                            return res_c, res_states
            except StopIteration:
                break

        if len(self.__statements) > 0:
            return self.__last_c, self.__statements


class NIFBufferedContextReader:
    def __init__(self, nif_path: str, window_size: int = 2000):
        self.data_name = os.path.basename(nif_path)
        self.buf = AutoPopBuffer(
            data_iter=ContextGroupedNIFReader(nif_path),
            default_value=[],
            window_size=window_size
        )

    def get(self, context: Union[rdflib.Graph, str]) -> List[state_type]:
        """
        We assume the order of querying keys is roughly the same as the order
        of keys in this data, that means we can find the key (context) within
        the current reading window. This is asymptotically similar to a full
        dataset search by increasing the window size.

        Args:
            context: The context to find in this window.

        Returns:
            A list of statements if found in the window, otherwise an empty
            list.

        """
        context_ = context_base(context) if isinstance(
            context, rdflib.Graph) else str(context)
        return self.buf.get_key(context_)


class AutoPopBuffer:
    def __init__(self, data_iter: Iterator, default_value,
                 window_size: int = 100):
        self.buf_data: OrderedDict[str, Tuple[Any, int]] = OrderedDict()

        self.__lookup_idx = 0
        self.__data_idx = 0

        self.__default_value = default_value
        self.__window_size = window_size
        self.__date_iter: Iterator = data_iter

    def get_key(self, key):
        value = self.__default_value

        self.__lookup_idx += 1

        if key in self.buf_data:
            value = self.buf_data.pop(key)[0]
        else:
            for k_, data in self.__date_iter:
                self.__data_idx += 1
                if k_ == key:
                    value = data
                else:
                    self.buf_data[k_] = (data, self.__data_idx)
                    if self.__data_idx - self.__lookup_idx > self.__window_size:
                        # Give up on this search.
                        break

        if len(self.buf_data) > 0:
            # Find the oldest index.
            _, (_, oldest_idx) = next(iter(self.buf_data.items()))
            # If the oldest index is out of the window, we will pop it.
            if 0 < oldest_idx < self.__lookup_idx - self.__window_size:
                self.buf_data.popitem(False)

        return value


def test_buf():
    # Test out the buffer reader.
    data_size = 10000
    buffer_size = 50

    data = range(0, data_size)
    data_new = [(d, d) for d in data]

    def swap_random(input_list, r):
        i = choice(range(len(input_list)))
        swap_range = choice(range(r))
        j = i + swap_range

        if j > len(input_list) - 1:
            return

        data_new[i], data_new[j] = data_new[j], data_new[i]
        print('swapped', i, j)

        if r > buffer_size:
            swapped[i] = j
            swapped[j] = i

    # Swap for some times.
    print('swapping in range')
    for _ in range(10):
        swap_random(data_new, buffer_size)

    print('swapping out of range')
    for _ in range(10):
        swap_random(data_new, buffer_size * 2)

    buf = AutoPopBuffer(iter(data_new), -1, buffer_size)

    for d in data:
        got = buf.get_key(d)
        print(got)


if __name__ == '__main__':
    swapped: Dict = {}
    test_buf()
