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
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import UtteranceContext


class TableReader(PackReader):
    def _collect(self, file_path: str) -> Iterator[str]:  # type: ignore
        with open(file_path) as f:
            for line in f:
                if line.startswith('Context:'):
                    yield line.split(':', 1)[1].strip()

    def _parse_pack(self, table: str) -> Iterator[DataPack]:
        p: DataPack = self.new_pack(pack_name='table_' + table.split("|")[0])
        p.set_text(table)

        # Create the table.
        UtteranceContext(p, 0, len(table))

        yield p
