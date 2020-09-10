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
The reader that reads AG News data into Datapacks.
"""
from typing import Iterator, Tuple

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.ag_news import Title, Description
from ft.onto.base_ontology import Document

__all__ = [
    "AGNewsReader",
]


class AGNewsReader(PackReader):
    r""":class:`AGNewsReaderr` is designed to read in AG News
    text classification dataset.
    """

    def __init__(self):
        super().__init__()
        self.configs = None

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable = unused-argument
        self.configs = configs

    def _collect(self, csv_file: str) -> Iterator[Tuple[int, str]]:
        r"""Collects from a CSV file path and returns an iterator of AG News
        data. The elements in the iterator correspond to each line
        in the csv file. One line is expected to be parsed as one
        DataPack.

        Args:
            csv_file: A CSV file path.

        Returns: Iterator of each line in the csv file.
        """
        with open(csv_file, "r") as f:
            for line_id, line in enumerate(f):
                yield (line_id, line)

    def _cache_key_function(self, line_info: Tuple[int, str]) -> str:
        return line_info[0]

    def _parse_pack(self, line_info: Tuple[int, str]) -> Iterator[DataPack]:
        line_id, line = line_info

        pack = self.new_pack()
        text: str = ""
        line = line.strip()
        data = line.split(",")

        class_id: int = int(data[0].replace("\"", ""))
        title: str = data[1]
        description: str = data[2]

        text += title
        title_end = len(text)
        text += "\n" + description
        description_start = title_end + 1
        description_end = len(text)

        pack.set_text(text, replace_func=self.text_replace_operation)

        doc = Document(pack, 0, description_end)
        doc.document_class = [self.configs.class_names[class_id]]
        Title(pack, 0, title_end)
        Description(pack, description_start, description_end)

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()

        config.update({
            'class_names': {
                1: 'World',
                2: 'Sports',
                3: 'Business',
                4: 'Sci/Tech'
            }
        })
        return config
