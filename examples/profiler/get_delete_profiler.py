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
from typing import List, Tuple
from time import time

from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Token, EntityMention, Document


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
        self.dataset_path = os.path.join(
            root_path, "Documents/forte/examples/profiler/combine_data"
        )

        # Define and config the Pipeline
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(OntonotesReader())
        self.nlp.add(DummyPackProcessor())
        self.nlp.set_profiling(True)
        self.nlp.initialize()
        self.data_pack: DataPack = self.nlp.process_one(self.dataset_path)

    def test_delete(self):
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            sentences = list(pack.get(Sentence))
            num_sent = len(sentences)
            first_sent = sentences[0]
            # delete first sentence
            pack.delete_entry(first_sent)
            self.assertEqual(len(list(pack.get_data(Sentence))), num_sent - 1)

    def test_get_request(self):
        # get processed pack from dataset
        get_timer = 0
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get data with required entry
            requests = {Sentence: ["speaker"], Token: ["pos", "sense"]}
            start_time = time()
            instances = list(
                pack.get_data(Sentence, request=requests, skip_k=1)
            )
            get_timer += time() - start_time
        print("get data with request: " + str(get_timer))
        self.nlp.finish()

    def test_get_data(self):
        # case 1: get sentence context from the beginning
        instances = list(self.data_pack.get_data(Sentence))
        self.assertEqual(len(instances), 20)
        self.assertEqual(
            instances[1]["offset"], len(instances[0]["context"]) + 1
        )

        # case 2: get sentence context from the second instance
        instances = list(self.data_pack.get_data(Sentence, skip_k=1))
        self.assertEqual(len(instances), 19)
        self.assertEqual(instances[0]["offset"], 44)

        # case 3: get document context
        instances = list(self.data_pack.get_data(Document, skip_k=0))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 0)

        # case 3.1: test get single
        document: Document = self.data_pack.get_single(Document)
        self.assertEqual(document.text, instances[0]["context"])

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data(Sentence, skip_k=20))
        self.assertEqual(len(instances), 0)

    def test_get_entries(self):
        # case 1: test get annotation
        sent_texts: List[Tuple[int, str]] = []
        for doc in self.data_pack.get(Document):
            for sent in self.data_pack.get(Sentence, doc):
                sent_texts.append(sent.text)
        self.assertEqual(
            sent_texts[0], "Powerful Tools for Biotechnology - Biochips"
        )


if __name__ == "__main__":
    unittest.main("get_delete_profiler")
