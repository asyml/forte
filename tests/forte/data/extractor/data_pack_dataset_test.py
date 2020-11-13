#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest

from typing import List

from forte.data.extractor.datapack_loader import DataPackLoader
from forte.data.data_pack import DataPack

from ft.onto.base_ontology import Sentence
from forte.data.readers.conll03_reader_new import CoNLL03Reader

from forte.data.extractor.data_pack_dataset import \
    DataPackDataSource, RawExample


class DataPackDatasetTest(unittest.TestCase):
    def setUp(self):
        file_path: str = "data_samples/data_pack_dataset_test"
        reader = CoNLL03Reader()
        context_type = Sentence
        request = {Sentence: []}
        skip_k = 0

        self.input_files = [
            "data_samples/data_pack_dataset_test/conll03_1.conll",
            "data_samples/data_pack_dataset_test/conll03_2.conll"
        ]
        self.feature_schemes = {}

        pack_loader: DataPackLoader = DataPackLoader(reader,
                                                     "foo",
                                                     {"cache": False,
                                                      "src_dir": file_path})

        self.data_source: DataPackDataSource = DataPackDataSource(pack_loader,
                                                                  context_type,
                                                                  request,
                                                                  skip_k)

    def test_data_pack_iterator(self):
        data_pack_iter = iter(self.data_source)
        raw_examples_1: List[RawExample] = []
        raw_examples_2: List[RawExample] = []

        for idx, raw_example in enumerate(data_pack_iter):
            curr_pack: DataPack = raw_example[1]
            if curr_pack.pack_name == self.input_files[0]:
                raw_examples_1.append(raw_example)
            else:
                raw_examples_2.append(raw_example)

        self.assertEqual(len(raw_examples_1), 7)
        self.assertEqual(len(raw_examples_2), 3)
        self.assertEqual(raw_examples_1[0][0]['context'],
                         "EU rejects German call to boycott British lamb .")
        self.assertEqual(raw_examples_1[1][0]['context'],
                         "Peter Blackburn")
        self.assertEqual(raw_examples_2[0][0]['context'],
                         "EU rejects German call to boycott British lamb .")
        self.assertEqual(raw_examples_2[1][0]['context'],
                         "Peter Blackburn")


if __name__ == '__main__':
    unittest.main()
