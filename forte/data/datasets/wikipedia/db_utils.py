"""
A set of utilities to support reading DBpedia datasets.
"""
import logging
import os
from typing import List, Dict, Tuple, Iterator
import bz2
import re
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

    def parse_graph(self, data: str, tuple_format: str) -> rdflib.Graph:
        if self.format == 'nquads':
            g_ = rdflib.ConjunctiveGraph()
        else:
            g_ = rdflib.Graph()

        g_.parse(data=data, format=tuple_format)

        if self.format == 'nquads':
            return g_.quads()
        else:
            return g_

    def read(self):
        while True:
            try:
                line = next(self.__nif)
                statements = self.parse_graph(
                    line.decode('utf-8'),
                    tuple_format=self.format
                )
                return list(statements)
            except StopIteration:
                break

        raise StopIteration

    def close(self):
        self.__nif.close()


class NIFBufferedContextReader:
    def __init__(self, nif_path: str, buffer_size: int = 100):
        self.data_name = os.path.basename(nif_path)

        self.__parser = NIFParser(nif_path)

        self.__buf_statement: Dict[str, List] = {}
        self.__buffer_size = buffer_size

        self.__last_c = ''
        self.__statements: List[state_type] = []

    def buf_info(self):
        logging.info('The buffer size for data [%s] is %s',
                     self.data_name, len(self.__buf_statement))

    def yield_by_context(self) -> Iterator[Tuple[str, List[state_type]]]:
        res_c: str = ''
        res_states: List = []

        while True:
            try:
                for statements in self.__parser:
                    for s, v, o, c in statements:
                        c_ = context_base(c)

                        if not c_ == self.__last_c and self.__last_c is not '':
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
            yield self.__last_c, self.__statements

    def get(self, context: rdflib.Graph) -> List[state_type]:
        # TODO: fix this.
        context_ = context_base(context)

        for c_, statements in self.yield_by_context():
            if c_ == context_:
                return statements
            elif context_ in self.__buf_statement:
                return self.__buf_statement[context_]
            elif self.__buffer_size > len(self.__buf_statement):
                self.__buf_statement[c_] = statements
            else:
                logging.info('[%s] not found in [%s]', context_, self.data_name)
