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
The reader that reads QA-SRL Bank 2.0 data into data pack.
Data Overview:
https://github.com/uwnlp/qasrl-bank
Data Format:
https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md
"""
import logging
import os
from typing import Iterator

from forte.common.exception import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Sentence, Document

__all__ = [
    "QASRLReader"
]


class QASRLReader(PackReader):
    r""":class:`QASRLReader` is designed to read
        in the QA-SRL Bank 2.0 dataset used
        by Question-Answer driven Semantic Role
        Labeling (QA-SRL) annotations task.
        The related paper can be found here
        https://arxiv.org/pdf/1805.05377.pdf
        The Original data format can be found it here:
        https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if configs.qa_file_extension is None:
            raise ProcessorConfigError(
                "Configuration qa_file_extension not provided.")

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""Iterator over text files in the data_source

        Args:
            args: args[0] is the directory to the .qa files.
            kwargs:

        Returns: Iterator over files in the path with qa extensions.
        """

        qa_directory: str = args[0]

        qa_file_extension: str = self.configs.qa_file_extension

        logging.info("Reading dataset from %s with extension %s",
                     qa_directory, qa_file_extension)
        return dataset_path_iterator(qa_directory, qa_file_extension)

    def _cache_key_function(self, qa_file: str) -> str:
        return os.path.basename(qa_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack: DataPack = self.new_pack()
        text: str = ""
        offset: int = 0

        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    sentence: str = line

                    # Add sentence.
                    Sentence(pack, offset, offset + len(sentence))
                    offset += len(sentence) + 1
                    text += sentence + " "

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        pack.pack_name = file_path

        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()
        # Add QA_SRL dataset file extension. The default is '.qa'
        config.update({'qa_file_extension': 'qa'})
        return config
