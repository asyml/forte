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
The reader that reads IMDB data into data pack.
Data Overview:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Data Format:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
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
from ft.onto.base_ontology import Sentence, Token, Document
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from forte.data.multi_pack import MultiPack


__all__ = [
    "IMDBReader"
]


class IMDBReader(PackReader):
    r""":class:`IMDBReader` is designed to read
        in the imdb review dataset used
        by sentiment classification task.
        The Original data format:
        "movie comment, positive"
        "movie comment, negative"
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if configs.imdb_file_extension is None:
            raise ProcessorConfigError(
                "Configuration imdb_file_extension not provided.")

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""Iterator over text files in the data_source

        Args:
            args: args[0] is the directory to the .imdb files.
            kwargs:

        Returns: Iterator over files in the path with imdb extensions.
        """

        imdb_directory: str = args[0]

        #imdb_file_extension: str = self.configs.imdb_file_extension
        imdb_file_extension = "imdb"

        logging.info("Reading dataset from %s with extension %s",
                     imdb_directory, imdb_file_extension)
        return dataset_path_iterator(imdb_directory, imdb_file_extension)

    def _cache_key_function(self, imdb_file: str) -> str:
        return os.path.basename(imdb_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack: DataPack = DataPack()
        text: str = ""
        offset: int = 0

        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    line_list = line.split("\",")
                    sentence = line_list[0].strip("\"")
                    sentiment = line_list[1]

                    # Add sentence.
                    senobj = Sentence(pack, offset+1,
                                      offset + len(sentence)+1)
                    #senobj.sentiment[sentence] = sentiment
                    #senobj.__setattr__("sentim" , sentiment)
                    senobj.speaker = sentiment

                    # Add token
                    wordoffset = offset+1
                    words = sentence.split(" ")
                    for word in words:
                        lastch = word[len(word)-1]
                        new_word = word
                        if lastch == "," or lastch=='.':
                            new_word = word[:len(word)-1]
                        Token(pack, wordoffset, wordoffset+len(new_word))
                        wordoffset += len(word)
                        # For space between words
                        wordoffset += 1

                    # For \n
                    offset += len(line) + 1
                    text += line + " "

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        pack.pack_name = file_path

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "kwargs": {
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            'type': 'data_augmentation_op',
            'data_aug_op': 'tests.forte.processors.base.data_augment_replacement_processor_test.TmpReplacer',
            "data_aug_op_config": {
                'kwargs': {}
            },
            'augment_pack_names': {
                'kwargs': {}
            }
        }

        processor = ReplacementDataAugmentProcessor()
        processor.initialize(resources=None, configs=processor_config)

        p = MultiPack()
        p.add_pack_(pack, "output_tgt")

        processor._process(p)
        new_tgt_pack= p.get_pack('augmented_output_tgt')

        yield pack
        yield new_tgt_pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()
        # Add imdb dataset file extension. The default is '.imdb'
        config.update({'imdb_file_extension': 'imdb'})
        return config
