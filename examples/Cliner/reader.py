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
The reader to read a table and example utterance.
"""
from typing import Iterator

from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document


class ClinerReader(PackReader):
    def _collect(self, txt_path: str) -> Iterator[str]:  # type: ignore
        yield txt_path

    def _parse_pack(self, collection) -> Iterator[DataPack]:
        txt_path = collection

        pack = DataPack(pack_name='Cliner_input')
        with open(txt_path, "r", encoding="utf-8") as doc_file:
            doc = doc_file.readlines()
            offsets = []
            offset = 0
            text = ""
            text_lines = []

            for line in doc:
                text += line.strip() + '\n'
                offsets.append(offset)  # the begin of the text
                offset += len(line)
                text_lines.append(line)

            pack.set_text(text, replace_func=self.text_replace_operation)

            Document(pack, 0, len(text))

            pack.pack_name = 'Cliner_input'

            yield pack
