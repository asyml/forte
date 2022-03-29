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
The reader that reads plain text data into Datapacks.
"""
import logging
from typing import Iterator, List, Union, Dict, Set

from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document

logger = logging.getLogger(__name__)

__all__ = [
    "StringReader",
]


class StringReader(PackReader):
    r"""
    :class:`StringReader` is designed to read in a list of string variables.
    """

    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return str(hash(collection)) + ".html"

    def _collect(  # type: ignore
        self, string_data: Union[List[str], str]
    ) -> Iterator[str]:
        r"""``string`_data` should be of type `List[str]`,
        which is the list of raw text strings to iterate over.
        """
        # This allows the user to pass in either one single string or a list of
        # strings.
        data_strings = (
            [string_data] if isinstance(string_data, str) else string_data
        )
        for data in data_strings:
            yield data

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        r"""Takes a raw string and converts into a :class:`~forte.data.data_pack.DataPack`.

        Args:
            data_source: str that contains text of a document.

        Returns: :class:`~forte.data.data_pack.DataPack` containing Document.
        """
        pack = DataPack()

        self.set_text(pack, data_source)
        Document(pack, 0, len(data_source))

        yield pack

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `StringReader` which is
        `ft.onto.base_ontology.Document` with an empty set
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Document"] = set()
