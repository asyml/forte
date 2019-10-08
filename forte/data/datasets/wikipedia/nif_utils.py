"""
A set of utilities to support reading DBpedia datasets.
"""
from typing import List

from urllib.parse import urlparse, parse_qs
import rdflib
import bz2
import re


def get_resource_attribute(url, param_name):
    parsed = urlparse(url)
    return parse_qs(parsed.query)[param_name][0]


def get_dbpedia_resource_name(url):
    parsed = urlparse(url)
    return re.sub('^/resource/', '', parsed.path)


def strip_url_params(url):
    parsed = urlparse(url)
    # scheme + netloc + path
    return parsed.scheme + "://" + parsed.netloc + parsed.path


class NIFContextReader:
    def __init__(self, nif_path):
        self.__parser = NIFParser(nif_path)

        self.__context: str = ''
        self.__statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.__parser.close()

    def __iter__(self):
        return self

    def __next__(self):
        for g in self.__parser:
            for statement in g:
                s, v, o, c = statement.quads()

                if not self.__context == c and not self.__context == c:
                    res = list(self.__statements)
                    self.__statements.clear()
                    return res

                self.__statements.append((s, v, o))
                self.__context = c


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

    def parse_graph(self, data: str, format: str) -> rdflib.Graph:
        if self.format == 'nquads':
            g_ = rdflib.ConjunctiveGraph()
        else:
            g_ = rdflib.Graph()

        g_.parse(data=data, format=format)
        return g_

    def read(self):
        while True:
            line = next(self.__nif)
            statements = self.parse_graph(
                line.decode('utf-8'),
                format=self.format
            )
            if statements:
                return list(statements)

    def close(self):
        self.__nif.close()
