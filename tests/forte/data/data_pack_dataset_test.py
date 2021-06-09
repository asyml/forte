#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
import os
import unittest
from typing import List, Iterator

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack_dataset import RawExample, DataPackIterator
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence


class DataPackDatasetTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )

        file_path: str = os.path.join(
            root_path, "data_samples/data_pack_dataset_test"
        )
        reader = CoNLL03Reader()
        context_type = Sentence
        request = {Sentence: []}
        skip_k = 0

        self.input_files = ["conll03_1.conll", "conll03_2.conll"]
        self.feature_schemes = {}

        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(reader)
        train_pl.initialize()
        pack_iterator: Iterator[PackType] = train_pl.process_dataset(file_path)

        self.data_source: DataPackIterator = DataPackIterator(
            pack_iterator, context_type, request, skip_k
        )

    def test_data_pack_iterator(self):
        data_pack_iter = iter(self.data_source)
        raw_examples_1: List[RawExample] = []
        raw_examples_2: List[RawExample] = []
        packs_1: List[DataPack] = []
        packs_2: List[DataPack] = []

        for idx, raw_example in enumerate(data_pack_iter):
            curr_pack: DataPack = raw_example[1]
            if curr_pack.pack_name == self.input_files[0]:
                raw_examples_1.append(raw_example)
                packs_1.append(curr_pack)
            else:
                raw_examples_2.append(raw_example)
                packs_2.append(curr_pack)

        self.assertEqual(len(raw_examples_1), 7)
        self.assertEqual(len(raw_examples_2), 3)
        self.assertEqual(
            packs_1[0].get_entry(raw_examples_1[0][0]).text,
            "EU rejects German call to boycott British lamb .",
        )
        self.assertEqual(
            packs_1[0].get_entry(raw_examples_1[1][0]).text, "Peter Blackburn"
        )
        self.assertEqual(
            packs_2[0].get_entry(raw_examples_2[0][0]).text,
            "EU rejects German call to boycott British lamb .",
        )
        self.assertEqual(
            packs_2[0].get_entry(raw_examples_2[1][0]).text, "Peter Blackburn"
        )


if __name__ == "__main__":
    unittest.main()
