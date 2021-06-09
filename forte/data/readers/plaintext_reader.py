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
import os
from typing import Any, Iterator, Dict, Set

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document

__all__ = [
    "PlainTextReader",
]


class PlainTextReader(PackReader):
    r""":class:`PlainTextReader` is designed to read in plain text dataset."""

    def _collect(self, text_directory) -> Iterator[Any]:  # type: ignore
        r"""Should be called with param ``text_directory`` which is a path to a
        folder containing txt files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .txt files
        """
        return dataset_path_iterator(text_directory, self.configs.file_ext)

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    # pylint: disable=unused-argument
    def text_replace_operation(self, text: str):
        return []

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = DataPack()

        with open(file_path, "r", encoding="utf8", errors="ignore") as file:
            text = file.read()

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(pack.text))

        pack.pack_name = file_path
        yield pack

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config["file_ext"] = ".txt"
        return config

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `PlainTextReader` which is
        `ft.onto.base_ontology.Document` with an empty set
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Document"] = set()
