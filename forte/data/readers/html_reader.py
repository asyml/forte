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
The reader that reads html data into Datapacks.
"""
from html.parser import HTMLParser
from html import unescape
import os
import re
from typing import Iterator

from forte.data.span import Span
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Document

# Regular expressions used for parsing. Borrowed from
# https://github.com/python/cpython/blob/3.6/Lib/html/parser.py

interesting_normal = re.compile('[&<]')
incomplete = re.compile('&[a-zA-Z#]')

entityref = re.compile('&([a-zA-Z][-.a-zA-Z0-9]*)[^a-zA-Z0-9]')
charref = re.compile('&#(?:[0-9]+|[xX][0-9a-fA-F]+)[^0-9a-fA-F]')

starttagopen = re.compile('<[a-zA-Z]')
piclose = re.compile('>')
commentclose = re.compile(r'--\s*>')
# Note:
#  1) if you change tagfind/attrfind remember to update locatestarttagend too;
#  2) if you change tagfind/attrfind and/or locatestarttagend the parser will
#     explode, so don't do it.
# see http://www.w3.org/TR/html5/tokenization.html#tag-open-state
# and http://www.w3.org/TR/html5/tokenization.html#tag-name-state
tagfind_tolerant = re.compile(r'([a-zA-Z][^\t\n\r\f />\x00]*)(?:\s|/(?!>))*')
attrfind_tolerant = re.compile(
    r'((?<=[\'"\s/])[^\s/>][^\s/=>]*)(\s*=+\s*'
    r'(\'[^\']*\'|"[^"]*"|(?![\'"])[^>\s]*))?(?:\s|/(?!>))*')
locatestarttagend_tolerant = re.compile(r"""
  <[a-zA-Z][^\t\n\r\f />\x00]*       # tag name
  (?:[\s/]*                          # optional whitespace before attribute name
    (?:(?<=['"\s/])[^\s/>][^\s/=>]*  # attribute name
      (?:\s*=+\s*                    # value indicator
        (?:'[^']*'                   # LITA-enclosed value
          |"[^"]*"                   # LIT-enclosed value
          |(?!['"])[^>\s]*           # bare value
         )
         (?:\s*,)*                   # possibly followed by a comma
       )?(?:\s|/(?!>))*
     )*
   )?
  \s*                                # trailing whitespace
""", re.VERBOSE)
endendtag = re.compile('>')
# the HTML 5 spec, section 8.1.2.2, doesn't allow spaces between
# </ and the tag name, so maybe this should be fixed
endtagfind = re.compile(r'</\s*([a-zA-Z][-.a-zA-Z0-9:_]*)\s*>')

__all__ = [
    "HTMLReader",
]


class ForteHTMLParser(HTMLParser):
    r"""Parser that stores spans that HTMLReader can use.
    """

    def __init__(self):
        super().__init__()
        self.spans = []

    def collect_span(self, begin, end):
        self.spans.append((Span(begin, end), ''))

    # We override the original goahead method and collect the information
    # we need to successfully remove tag information and retrieve the original
    # html document without any loss.
    def goahead(self, end):
        rawdata = self.rawdata
        i = 0
        n = len(rawdata)
        while i < n:
            if self.convert_charrefs and not self.cdata_elem:
                j = rawdata.find('<', i)
                if j < 0:
                    # if we can't find the next <, either we are at the end
                    # or there's more text incoming.  If the latter is True,
                    # we can't pass the text to handle_data in case we have
                    # a charref cut in half at end.  Try to determine if
                    # this is the case before proceeding by looking for an
                    # & near the end and see if it's followed by a space or ;.
                    amppos = rawdata.rfind('&', max(i, n - 34))
                    if (amppos >= 0 and
                            not re.compile(r'[\s;]').search(rawdata, amppos)):
                        break  # wait till we get all the text
                    j = n
            else:
                match = self.interesting.search(rawdata, i)  # < or &
                if match:
                    j = match.start()
                else:
                    if self.cdata_elem:
                        break
                    j = n
            if i < j:
                if self.convert_charrefs and not self.cdata_elem:
                    self.handle_data(unescape(rawdata[i:j]))
                else:
                    self.handle_data(rawdata[i:j])
            i = self.updatepos(i, j)
            if i == n:
                break
            startswith = rawdata.startswith
            if startswith('<', i):
                if starttagopen.match(rawdata, i):  # < + letter
                    k = self.parse_starttag(i)
                    self.collect_span(i, k)
                elif startswith("</", i):
                    k = self.parse_endtag(i)
                    self.collect_span(i, k)
                elif startswith("<!--", i):
                    k = self.parse_comment(i)
                    self.collect_span(i, k)
                elif startswith("<?", i):
                    k = self.parse_pi(i)
                    self.collect_span(i, k)
                elif startswith("<!", i):
                    k = self.parse_html_declaration(i)
                    self.collect_span(i, k)
                elif (i + 1) < n:
                    self.handle_data("<")
                    k = i + 1
                else:
                    break
                if k < 0:
                    if not end:
                        break
                    k = rawdata.find('>', i + 1)
                    if k < 0:
                        k = rawdata.find('<', i + 1)
                        if k < 0:
                            k = i + 1
                    else:
                        k += 1
                    if self.convert_charrefs and not self.cdata_elem:
                        self.handle_data(unescape(rawdata[i:k]))
                    else:
                        self.handle_data(rawdata[i:k])
                i = self.updatepos(i, k)
            elif startswith("&#", i):
                match = charref.match(rawdata, i)
                if match:
                    name = match.group()[2:-1]
                    self.handle_charref(name)
                    k = match.end()
                    if not startswith(';', k - 1):
                        k = k - 1
                    i = self.updatepos(i, k)
                else:
                    if ";" in rawdata[i:]:  # bail by consuming &#
                        self.handle_data(rawdata[i:i + 2])
                        i = self.updatepos(i, i + 2)
                    break
            elif startswith('&', i):
                match = entityref.match(rawdata, i)
                if match:
                    name = match.group(1)
                    self.handle_entityref(name)
                    k = match.end()
                    if not startswith(';', k - 1):
                        k = k - 1
                    i = self.updatepos(i, k)
                    continue
                match = incomplete.match(rawdata, i)
                if match:
                    # match.group() will contain at least 2 chars
                    if end and match.group() == rawdata[i:]:
                        k = match.end()
                        if k <= i:
                            k = n
                        i = self.updatepos(i, i + 1)
                    # incomplete
                elif (i + 1) < n:
                    # not the end of the buffer, and can't be confused
                    # with some other construct
                    self.handle_data("&")
                    i = self.updatepos(i, i + 1)
                else:
                    break
            else:
                assert 0, "interesting.search() lied"
        # end while
        if end and i < n and not self.cdata_elem:
            if self.convert_charrefs and not self.cdata_elem:
                self.handle_data(unescape(rawdata[i:n]))
            else:
                self.handle_data(rawdata[i:n])
            i = self.updatepos(i, n)
        # pylint: disable=attribute-defined-outside-init
        self.rawdata = rawdata[i:]


class HTMLReader(PackReader):
    r""":class:`HTMLReader` is designed to read in list of html strings.

    It takes in list of html strings, cleans the HTML tags and stores the
    cleaned text in pack.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_with_fileloc = False
        self.init_with_html = False

    def _collect(self, content) -> Iterator[str]:  # type: ignore
        r"""Could be called with a directory, a particular file location or a
        list of strings. If the string is an HTML string, it will be cleaned.

        Args:
            content: either a string, or list of string

        Returns: Iterator over the content based on type of input
        """
        if isinstance(content, str):
            # Check if directory
            if os.path.isdir(content):
                self.init_with_fileloc = True
                # TODO: maybe extend it to .txt also if need be?
                return dataset_path_iterator(content, ".html")
            # If file path to a single file, just return the filepath
            elif os.path.isfile(content):
                def data_yielder(data):
                    yield data

                self.init_with_fileloc = True
                return data_yielder(content)
            else:  # Treat it as a string
                content = [content]

        if isinstance(content, list):  # Must be a list of strings now
            self.init_with_html = True

            def data_iterator(data):
                for html_string in data:
                    yield html_string

            return data_iterator(content)

        else:
            raise TypeError(f"HTMLReader supports only strings and list of"
                            f" strings, Please make sure your inputs are"
                            f" correct!"
                            f"Found {type(content)} instead!")

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        r"""Takes a string which could be either a filepath or html_content and
        converts into a DataPack.

        Args:
            data_source: str that contains text of a document or a filepath

        Returns: DataPack containing Document.
        """
        pack = self.new_pack()

        # Check if data_source is a filepath
        if self.init_with_fileloc:
            with open(data_source, "r",
                      encoding="utf8",
                      errors='ignore') as file:
                text = file.read()
        # else, must be a string with actual data
        else:
            text = data_source

        self.set_text(pack, text)
        # Note that pack.text can be different from the text passed in, due to
        # the text_replace_operation
        Document(pack, 0, len(pack.text))

        yield pack

    def text_replace_operation(self, text: str):
        r"""Replace html tag locations with blank string.

        Args:
            text: The original html text to be cleaned.

        Returns: List[Tuple[Span, str]]: the replacement operations

        """
        parser = ForteHTMLParser()
        parser.feed(text)

        return parser.spans

    def _cache_key_function(self, collection):
        # check if collection is file or html string
        if self.init_with_fileloc:
            return os.path.basename(collection)
        # If html string
        else:
            return str(hash(collection)) + '.html'
