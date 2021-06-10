# Copyright 2020 The Forte Authors. All Rights Reserved.
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
A reader to read reviews and IMDb scores from
`Large Movie Reviews` dataset for sentiment classification.

There are two directories [pos/, neg/] for the reviews.
When using this reader, pass in 'data_dir/pos' and 'data_dir/neg',
respectively, as the file_path, to generate datapacks.

Dataset download link -
https://ai.stanford.edu/~amaas/data/sentiment/
Dataset Paper Citation -
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.
                and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for
                    Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

import os
import logging
import re
from typing import Iterator, List

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document, Sentence

__all__ = ["LargeMovieReader"]


class LargeMovieReader(PackReader):
    r""":class:`LargeMovieReader` is designed to read in
    the Large Movie Review Dataset v1.0.
    Reviews are stored in text files named
    following the convention [[id]_[rating].txt].
    """

    def __init__(self):
        super().__init__()
        self.REPLACE_NO_SPACE = re.compile(
            r"(\:)|(\')|(\,)|(\")|(\()|(\))|(\[)|(\])"
        )
        self.REPLACE_WITH_NEWLINE = re.compile(
            r"(<br\s*/><br\s*/>)|(\-)|(\/)|(\.)|(\;)|(\!)|(\?)"
        )

    def preprocess_reviews(self, para):
        para = self.REPLACE_NO_SPACE.sub("", para.lower())
        para = self.REPLACE_WITH_NEWLINE.sub("\n", para)
        return para

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""Iterator over text files in the data_source.

        Args:
            args: args[0] is the directory to the pos/neg movie files.
            kwargs:

        Returns: Iterator over files in the path with txt extensions.
        """
        movie_directory: str = args[0]
        logging.info("Reading .txt from %s", movie_directory)
        return dataset_path_iterator(movie_directory, "txt")

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        data_pack: DataPack = DataPack()

        sent_begin: int = 0
        doc_text: str = ""

        with open(file_path, encoding="utf8") as doc:
            for para in doc:
                para = self.preprocess_reviews(para)
                sents = para.split("\n")
                for sent in sents:
                    if len(sent) > 0:
                        sent = sent.strip()
                        doc_text += sent + " "
                        doc_offset = sent_begin + len(sent) + 1
                        # Add sentences.
                        Sentence(data_pack, sent_begin, doc_offset - 1)
                        sent_begin = doc_offset

        pos_dir: str = os.path.basename(os.path.dirname(file_path))
        movie_file: str = os.path.basename(file_path)
        title: List = movie_file.split("_")
        doc_id: str = pos_dir + title[0]
        score: float = float(title[1].split(".")[0])
        score /= 10.0

        data_pack.pack_name = doc_id
        data_pack.set_text(doc_text)

        # Add documents.
        document: Document = Document(data_pack, 0, len(doc_text))
        document.sentiment = {doc_id: score}

        yield data_pack

    def _cache_key_function(self, movie_file: str) -> str:
        return os.path.basename(os.path.dirname(movie_file)) + os.path.basename(
            movie_file
        )
