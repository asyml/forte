# Copyright 2021 The Forte Authors. All Rights Reserved.
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

import csv
import logging
from pathlib import Path
from typing import Any, Iterator, Union, List

from smart_open import open

from demo.clinical import Description, Body
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader


class Mimic3DischargeNoteReader(PackReader):
    """This class is designed to read the discharge notes from MIMIC3 dataset
    as plain text packs.

    For more information for the dataset, visit:
      https://mimic.physionet.org/
    """

    def __init__(self):
        super().__init__()
        self.headers: List[str] = []
        self.text_col = -1  # Default to be last column.
        self.description_col = 0  # Default to be first column.
        self.__note_count = 0  # Count number of notes processed.

    def _collect(self, mimic3_path: Union[Path, str]) -> Iterator[Any]:
        with open(mimic3_path) as f:
            for r in csv.reader(f):
                if 0 < self.configs.max_num_notes <= self.__note_count:
                    break
                yield r

    def _parse_pack(self, row: List[str]) -> Iterator[DataPack]:
        if len(self.headers) == 0:
            self.headers.extend(row)
            for i, h in enumerate(self.headers):
                if h == 'TEXT':
                    self.text_col = i
                    logging.info("Text Column is %d", i)
                if h == 'DESCRIPTION':
                    self.description_col = i
                    logging.info("Description Column is %d", i)
        else:
            pack: DataPack = DataPack()
            description: str = row[self.description_col]
            text: str = row[self.text_col]
            delimiter = '\n-----------------\n'
            full_text = description + delimiter + text
            pack.set_text(full_text)

            Description(pack, 0, len(description))
            Body(pack, len(description) + len(delimiter), len(full_text))
            self.__note_count += 1
            yield pack

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        # If this is set (>0), the reader will only read up to
        # the number specified.
        config['max_num_notes'] = -1
        return config
