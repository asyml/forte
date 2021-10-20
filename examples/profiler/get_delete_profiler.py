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

import os
import unittest
from typing import Tuple, List

from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence

class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        pass


class OntonoteGetterPipelineTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
        # Define and config the Pipeline
        self.dataset_path = os.path.join(root_path, "Documents/forte/examples/profiler/combine_data")
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(OntonotesReader())

        self.nlp.add(DummyPackProcessor())
        
        self.nlp.initialize()

    def test_process_delete(self):
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            sentences = list(pack.get(Sentence))
            num_sent = len(sentences)
            first_sent = sentences[0]
            pack.delete_entry(first_sent)
            self.assertEqual(
                len(list(pack.get_data(Sentence))), num_sent - 1)


if __name__ == "__main__":
    unittest.main('get_delete_profiler')
