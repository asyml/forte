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

from typing import Iterator, Tuple

from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import UtteranceContext, Utterance


class TableReader(PackReader):
    def _collect(self, table: str, sentence: str  # type: ignore
                 ) -> Iterator[Tuple[str, str]]:
        yield table, sentence

    def _parse_pack(self, collection) -> Iterator[DataPack]:
        table, sentence = collection

        p: DataPack = self.new_pack(pack_name='rewriting_input')
        p.set_text(table + '\n' + sentence)

        # Create the table.
        UtteranceContext(p, 0, len(table))

        # Create the sample sentence.
        u = Utterance(p, len(table) + 1, len(p.text))
        u.speaker = 'ai'

        yield p
