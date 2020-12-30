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
        self.doc_ids: Dict[str, Tuple[str, str]] = \
            {"pos": ("pos3", "pos0"),
             "neg": ("neg3", "neg1")}
        # pos0 doc's leading text, neg1 doc's ending text.
        self.doc_text: Dict[str, str] = \
        {"pos":
            'Bromwell High is a cartoon comedy. It ran at the same time'
            ' as some other programs about school life, such as "Teache'
            'rs". My 35 years in the teaching profession lead me to bel'
            'ieve that Bromwell High\'s satire is much closer to reality '
            'than is "Teachers". The scramble to survive financially, the'
            ' insightful students who can see right through their pathetic'
            ' teachers\' pomp, the pettiness of the whole situation, all '
            'remind me of the schools I knew and their students. When I saw'
            ' the episode in which a student repeatedly tried to burn down '
            'the school, I immediately recalled ......... at .......... '
            'High. A classic line: INSPECTOR: I\'m here to sack one of your '
            'teachers. STUDENT: Welcome to Bromwell High. I expect that '
            'many adults of my age think that Bromwell High is far fetched.'
            ' What a pity that it isn\'t!',
         "neg":
             'Robert DeNiro plays the most unbelievably intelligent '
             'illiterate of all time. This movie is so wasteful of talent,'
             ' it is truly disgusting. The script is unbelievable. The '
             'dialog is unbelievable. Jane Fonda\'s character is a caricature'
             ' of herself, and not a funny one. The movie moves at a snail\'s'
             ' pace, is photographed in an ill-advised manner, and is '
             'insufferably preachy. It also plugs in every cliche in the '
             'book. Swoozie Kurtz is excellent in a supporting role, but so'
             ' what?  Equally annoying is this new IMDB rule of requiring '
             'ten lines for every review. When a movie is this worthless, '
             'it doesn\'t require ten lines of text to let other readers '
             'know that it is a waste of time and tape. Avoid this movie.'}
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
                # Test doc_ids.
                self.assertTrue(pack.pack_name in self.doc_ids[dir])

                # Test documents.
                docid0 = self.doc_ids[dir][0]
                docid1 = self.doc_ids[dir][1]
                if pack.pack_name == docid1:
                    for doc in pack.get(Document):
                        print(doc.text)
                        self.assertIn(self.doc_text[dir], doc.text)
                        # test sentiments.
                        self.assertEqual(
                            doc.sentiment[docid1], self.score[dir])

                count_packs += 1

            self.assertEqual(count_packs, 2)


if __name__ == "__main__":
    unittest.main()
