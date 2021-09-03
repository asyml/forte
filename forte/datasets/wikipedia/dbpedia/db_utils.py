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
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import rdflib

dbpedia_prefix = "http://dbpedia.org/resource/"
state_type = Tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]


def load_redirects(redirect_path: str) -> Dict[str, str]:
    # pylint: disable=line-too-long
    """
    Loads the Wikipedia link redirects as a Dictionary. The key is the page
    directed from, and the value is the page being directed to.

    Args:
        redirect_path (str): A path pointing to a file that contains the NIF
          formatted statements of redirect. A file like this can be obtained
          here: http://wiki.dbpedia.org/services-resources/documentation/datasets#Redirects

    Returns (dict):
        The dictionary containing the redirect information, pointing from
        key to value.
    """

    redirect_to = {}
    redirect_rel = "http://dbpedia.org/ontology/wikiPageRedirects"

    count = 0
    for statements in NIFParser(redirect_path):
        for statement in statements:
            s, v, o, _ = statement
            if str(v) == redirect_rel:
                from_page = get_resource_name(s)
                redirect_page = get_resource_name(o)
                if from_page is not None and redirect_page is not None:
                    redirect_to[from_page] = redirect_page
                    count += 1
                if count % 50000 == 0:
                    logging.info("Loaded %d redirects.", count)
    return redirect_to


def get_resource_attribute(url: str, param_name: str) -> Optional[str]:
    # pylint: disable=line-too-long
    """
    A utility function that extract the attribute of the resource from a NIF
    URL.

    >>> sample_url = 'http://dbpedia.org/resource/Animalia_(book)?dbpv=2016-10&nif=context'
    >>> get_resource_attribute(sample_url, 'nif')
    'context'

    Args:
        url (str): A NIF URL.
        param_name (str): The attribute name to extract.

    Returns (str):
        The extracted parameter value.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        logging.warning("Encounter un-parsable URL [%s]", url)
        return None

    return parse_qs(parsed.query)[param_name][0]


def context_base(c: rdflib.Graph) -> Optional[str]:
    """
    Take the base URL (context) from an URI from an statement.

    Args:
        c: The statement (which is a parsed rdflib.Graph object)

    Returns:
        The base URL. None if the URL cannot be parsed.

    """
    return strip_url_params(c.identifier)


def get_resource_fragment(url: str) -> Optional[str]:
    # pylint: disable=line-too-long
    """
    Get the resource fragment from an URL.

    >>> sample_url = 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString'
    >>> get_resource_fragment(sample_url)
    'isString'

    Args:
        url: The URL to find resource.

    Returns:
        The resource fragment.
    """
    try:
        return urlparse(url).fragment
    except ValueError:
        logging.warning("Encounter un-parsable URL [%s]", url)
        return None


def get_resource_name(
    url: str, resource_domain="http://dbpedia.org/resource"
) -> Optional[str]:
    # pylint: disable=line-too-long
    """
    Get the name of the resource from the URL.

    >>> get_resource_name("http://dbpedia.org/resource/Animalia_(book)?dbpv=2016-10&nif=context")
    'Animalia_(book)'

    >>> get_resource_name("http://dbpedia.org/resource/A_grave")
    'A_grave'

    # Handling wierd input from DBpedia dumps.
    >>> get_resource_name("http://dbpedia.org/resource/A;sldkfj")
    'A;sldkfj'

    >>> get_resource_name("http://dbpedia.org/resource/A;sldkfj?nif=context")
    'A;sldkfj'

    Args:
        url: The URL to find the resource name. None if the URL cannot be
          correctly parsed.

    Returns:
        The resource name.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        logging.warning("Encounter un-parsable URL [%s]", url)
        return None

    if url.startswith(resource_domain):
        reconstruct = parsed.path
        if not parsed.params == "":
            reconstruct = parsed.path + ";" + parsed.params
            # Sometimes there are ill-formed URL or resource name.
            if reconstruct not in url:
                logging.warning(
                    "Encounter unexpected resource URL [%s]. This resource "
                    "name may contain unexpected characters",
                    url,
                )
                return None

        # Params of the last fragment seem to be needed.
        return re.sub("^/resource/", "", reconstruct)
    else:
        return str(url)


def strip_url_params(url) -> Optional[str]:
    # pylint: disable=line-too-long
    """
    Take only the base URL and strip the parameters.

    >>> sample_url = 'http://dbpedia.org/resource/Animalia_(book)?dbpv=2016-10&nif=context'
    >>> strip_url_params(sample_url)
    'http://dbpedia.org/resource/Animalia_(book)'

    Args:
        url: The URL to strip.

    Returns:
        The base URL without all parameters.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        logging.warning("Encounter un-parsable URL [%s]", url)
        return None

    return parsed.scheme + "://" + parsed.netloc + parsed.path


def print_progress(msg: str, end="\r", terminal_only=False):
    """
    Print progress message to the same line.
    Args:
        msg: The message to print.
        end: Line ending in terminal.
        terminal_only: If True, will only write to terminal. Default is False.

    """
    if not terminal_only:
        logging.info(msg)
    sys.stdout.write("\033[K")  # Clear to the end of line.
    print(f" -- {msg}", end=end)


def print_notice(msg: str):
    """
    Print additional notice in a new line.

    Args:
        msg: The message to print.

    """
    print(f"\n -- {msg}")


class NIFParser:
    """
    This is a simple Parser that reads NIF tuples into list of statements. The
    parser can be used as context manager.

    .. code-block:: python
        for statements in NIFParser(some_path):
            # do something with the statements.

    """

    def __init__(self, nif_path: str, tuple_format="nquads"):
        """

        Args:
            nif_path:
            tuple_format:
        """
        # pylint: disable=consider-using-with
        if nif_path.endswith(".bz2"):
            self.__nif = bz2.BZ2File(nif_path)
        else:
            self.__nif = open(nif_path, "rb")  # type: ignore

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
        if self.format == "nquads":
            g_ = rdflib.ConjunctiveGraph()
        else:
            g_ = rdflib.Graph()

        g_.parse(data=data, format=tuple_format)

        if self.format == "nquads":
            return list(g_.quads())
        else:
            return list(g_)

    def read(self):
        while True:
            line = next(self.__nif)
            statements = list(
                self.parse_graph(line.decode("utf-8"), tuple_format=self.format)
            )

            if len(statements) > 0:
                return list(statements)

    def close(self):
        self.__nif.close()


class ContextGroupedNIFReader:
    """
    This reader parses the NIF tuples into statements, and group the results
    by the Subject of the statements. This parser can be used as a context
    manager.

    .. code-block:: python
        for subject, statements in ContextGroupedNIFReader(nif_path):
            # Do something with the results.
    """

    def __init__(self, nif_path: str):
        self.__parser = NIFParser(nif_path)
        self.data_name = os.path.basename(nif_path)

        self.__last_c: str = ""
        self.__statements: List[state_type] = []
        self.__finished: bool = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__parser.close()

    def __iter__(self):
        return self

    def __next__(self):
        res_c: str = ""
        res_states: List = []

        while True:
            try:
                # Call the NIF parser, but grouped the statements with
                # the same context.
                statements = next(self.__parser)
                for s, v, o, c in statements:
                    c_ = context_base(c)

                    if c_ is None:
                        continue

                    if c_ != self.__last_c and self.__last_c != "":
                        res_c = self.__last_c
                        res_states.extend(self.__statements)
                        self.__statements.clear()

                    self.__statements.append((s, v, o))
                    self.__last_c = c_

                    if not res_c == "":
                        return res_c, res_states
            except StopIteration:
                break

        # Remember to flush out the last bit.
        if not self.__finished and len(self.__statements) > 0:
            self.__finished = True
            return self.__last_c, self.__statements

        raise StopIteration
