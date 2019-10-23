"""
A set of utilities to support reading DBpedia datasets.
"""
import logging
import os
import sys
from typing import List, Dict, Tuple, Union
import bz2
import re
from urllib.parse import urlparse, parse_qs
from collections import OrderedDict

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


def print_progress(msg: str):
    """
    Print progress message to the same line.
    Args:
        msg: The message to print

    Returns:

    """
    sys.stdout.write("\033[K")  # Clear to the end of line.
    print(f' -- {msg}', end='\r')


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
    def __init__(self, nif_path: str, buffer_size: int = 500):
        self.data_name = os.path.basename(nif_path)

        self.__parser = ContextGroupedNIFReader(nif_path)

        self.window_statement: OrderedDict[
            str, Tuple[List, int]] = OrderedDict()
        self.__buffer_size = buffer_size
        self.__entry_index = 0

    def window_info(self):
        logging.info('The buffer size for data [%s] is %s',
                     self.data_name, len(self.window_statement))

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

        if context_ in self.window_statement:
            return self.window_statement.pop(context_)[0]

        for c_, statements in self.__parser:
            self.__entry_index += 1
            if c_ == context_:
                return statements
            else:
                self.window_statement[c_] = (statements, self.__entry_index)

                # Find the oldest index.
                oldest_index = -1
                for _, (_, index) in self.window_statement.items():
                    oldest_index = index
                    break

                # If the oldest index is out of the window, we will pop it.
                if 0 < oldest_index <= self.__entry_index - self.__buffer_size:
                    self.window_statement.popitem(False)

                if len(self.window_statement) >= self.__buffer_size:
                    # Give up on this search.
                    return []

        return []
