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
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Sentence, Token, Document


__all__ = [
    "IMDBReader"
]

logger = logging.getLogger(__name__)


class IMDBReader(PackReader):
    r""":class:`IMDBReader` is designed to read
        in the imdb review dataset used
        by sentiment classification task.
        The Original data format:
        "movie comment, positive"
        "movie comment, negative"
    """

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        r"""Iterator over text files in the data_source

        Args:
            args: args[0] is the directory to the .imdb files.
            kwargs:

        Returns: Iterator over files in the path with imdb extensions.
        """

        imdb_directory: str = args[0]

        imdb_file_extension = "imdb"

        logger.info(type(kwargs))


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
                    senobj = Sentence(pack, offset + 1,
                                      offset + len(sentence) + 1)
                    senobj.speaker = sentiment

                    # Add token
                    wordoffset = offset + 1
                    words = sentence.split(" ")
                    for word in words:
                        lastch = word[len(word) - 1]
                        new_word = word
                        if lastch in (',', '.'):
                            new_word = word[:len(word) - 1]
                        Token(pack, wordoffset, wordoffset + len(new_word))
                        wordoffset += len(word)
                        # For space between words
                        wordoffset += 1

                    # For \n
                    offset += len(line) + 1
                    text += line + " "

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        pack.pack_name = file_path

        yield pack
