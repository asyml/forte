"""
A set of utilities to support reading DBpedia datasets.
"""
import logging
import os
from typing import List, Dict

from urllib.parse import urlparse, parse_qs
import rdflib
import bz2
import re

dbpedia_prefix = "http://dbpedia.org/resource/"


def load_redirects(redirect_path: str) -> Dict[str, str]:
    redirect_to = {}
    redirect_rel = "http://dbpedia.org/ontology/wikiPageRedirects"

    count = 0
    for statements in NIFParser(redirect_path):
        for statement in statements:
            s, v, o, c = statement
            if str(v) == redirect_rel:
                count += 1
                from_page = get_resource_name(s)
                redirect_page = get_resource_name(o)
                redirect_to[from_page] = redirect_page
    return redirect_to


def get_resource_attribute(url, param_name) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query)[param_name][0]


def get_resource_fragment(url) -> str:
    return urlparse(url).fragment


def get_resource_name(url) -> str:
    parsed = urlparse(url)
    return re.sub('^/resource/', '', parsed.path)


def strip_url_params(url) -> str:
    parsed = urlparse(url)
    # scheme + netloc + path
    return parsed.scheme + "://" + parsed.netloc + parsed.path


class NIFBufferedContextReader:
    def __init__(self, nif_path: str, buffer_size: int = 100):
        self.data_name = os.path.basename(nif_path)

        self.__parser = NIFParser(nif_path)

        self.__buf_statement: Dict[str, List] = {}
        self.__prev_context: str = ''
        self.__buffer_size = buffer_size

    def buf_info(self):
        logging.info(f'The buffer size for data [{self.data_name}] '
                     f'is {len(self.__buf_statement)}')


    def get(self, context: rdflib.Graph):
        context_ = strip_url_params(context.identifier)

        if context_ in self.__buf_statement:
            return self.__buf_statement.pop(context_)
        else:
            # TODO: check if iterator will continue with the new for.
            statements = []

            for g in self.__parser:
                if len(self.__buf_statement) >= self.__buffer_size:
                    return []

                for s, v, o, c in g:
                    c_ = strip_url_params(c.identifier)
                    if c_ not in self.__buf_statement:
                        # Read in new contexts to the buffer.
                        self.__buf_statement[c_] = [(s, v, o)]

                        if self.__prev_context == context_:
                            # If a previous context is as required.
                            statements = self.__buf_statement.pop(context_)
                        self.__prev_context = c_

                        if statements:
                            return statements
                    else:
                        # Context already in buffer.
                        self.__buf_statement[c_].append((s, v, o))


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
            line = next(self.__nif)
            statements = self.parse_graph(
                line.decode('utf-8'),
                tuple_format=self.format
            )
            if statements:
                return list(statements)

    def close(self):
        self.__nif.close()
