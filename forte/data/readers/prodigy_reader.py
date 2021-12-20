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
The reader that reads prodigy text data with annotations into Datapacks.
"""

import json
from typing import Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Token, Document, EntityMention

__all__ = ["ProdigyReader"]


class ProdigyReader(PackReader):
    r""":class:`ProdigyReader` is designed to read in Prodigy output text."""

    def _cache_key_function(self, data: dict) -> str:
        return data["meta"]["id"]

    def _collect(  # type: ignore
        self, prodigy_annotation_file: str
    ) -> Iterator[Any]:
        r"""Collects from Prodigy file path and returns an iterator of Prodigy
        annotation data. The elements in the iterator correspond to each line
        in the prodigy file. One element is expected to be parsed as one
        DataPack.

        Args:
            prodigy_annotation_file: A Prodigy file path.

        Returns: Iterator of each line in the prodigy file.
        """
        with open(prodigy_annotation_file, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def _parse_pack(self, data: dict) -> Iterator[DataPack]:
        r"""Extracts information from input `data` of one document output from
        Prodigy Annotator including the text, tokens and its annotations into a
        DataPack.

        Args:
            data: a dict that contains information for one document.

        Returns: DataPack containing information extracted from `data`.
        """
        pack = DataPack()
        text = data["text"]
        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        tokens = data["tokens"]
        spans = data["spans"]
        for token in tokens:
            begin = token["start"]
            end = token["end"]
            Token(pack, begin, end)

        for span_items in spans:
            begin = span_items["start"]
            end = span_items["end"]
            annotation_entry = EntityMention(pack, begin, end)
            annotation_entry.ner_type = span_items["label"]

        pack.pack_name = data["meta"]["id"]

        yield pack
