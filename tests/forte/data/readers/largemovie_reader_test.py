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
Unit tests for LargeMovieReader.
"""

import os
import unittest
from typing import Iterable, Dict, Tuple

from forte.data.data_pack import DataPack
from forte.data.readers import LargeMovieReader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Document, Sentence


class LargeMovieReaderTest(unittest.TestCase):

    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        movie_dir = 'data_samples/large_movie_review'
        self.dataset_path: Dict[str, str] = \
            {"pos": os.path.abspath(os.path.join(file_dir_path,
                                                 *([os.path.pardir] * 4),
                                                 movie_dir, 'pos')),
             "neg": os.path.abspath(os.path.join(file_dir_path,
                                                 *([os.path.pardir] * 4),
                                                 movie_dir, 'neg'))}
        self.doc_ids: Dict[str,Tuple[str, str]] = \
            {"pos": ("pos3", "pos0"),
             "neg": ("neg3", "neg1")}
        # pos0 doc's leading text, neg1 doc's ending text.
        self.doc_text: Dict[str, str] = \
            {"pos": "bromwell high is a cartoon comedy it ran at the same time as some other programs "
                    "about school life such as teachers my 35 years in the teaching profession "
                    "lead me to believe that bromwell highs satire is much closer to reality than is teachers",
             "neg": "this new imdb rule of requiring ten lines for every review when a movie is "
                    "this worthless it doesnt require ten lines of text to let other readers know that it is "
                    "a waste of time and tape avoid this movie"}
        # pos3 sentence #1, neg3 sentence #5.
        self.sent_text: Dict[str, str] = \
            {"pos": "all the worlds a stage and its people actors in it",
             "neg": "i was put through tears repulsion shock anger sympathy and misery "
                    "when reading about the women of union street"}
        # pos0 doc's score, neg1 doc's score.
        self.score: Dict[str, float] = \
            {"pos": 0.9,
             "neg": 0.1}

        self.pipeline = Pipeline()
        reader = LargeMovieReader()
        self.pipeline.set_reader(reader)
        self.pipeline.initialize()

    def test_reader_text(self):
        for dir in ["pos", "neg"]:
            data_packs: Iterable[DataPack] = self.pipeline.process_dataset(
                self.dataset_path[dir])

            count_packs = 0
            for pack in data_packs:
                # test doc_ids.
                self.assertTrue(pack.pack_name in self.doc_ids[dir])

                # test documents.
                docid0 = self.doc_ids[dir][0]
                docid1 = self.doc_ids[dir][1]
                if pack.pack_name == docid1:
                    for doc in pack.get(Document):
                        self.assertIn(self.doc_text[dir], doc.text)
                        # test sentiments.
                        self.assertEqual(doc.sentiment[docid1], self.score[dir])
                # test sentences.
                elif pack.pack_name == docid0:
                    sents = pack.get(Sentence)
                    self.assertTrue(self.sent_text[dir] in [sent.text for sent in sents])

                count_packs += 1

            self.assertEqual(count_packs, 2)


if __name__ == "__main__":
    unittest.main()
