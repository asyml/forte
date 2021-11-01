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
import time
import os
import unittest
from typing import List

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
            )
        )
        self.dataset_path = os.path.join(
            root_path, "data_samples/profiler/combine_data"
        )

        # Define and config the Pipeline
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(OntonotesReader())
        self.nlp.add(DummyPackProcessor())
        self.nlp.initialize()

    def test_get_delete(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2-t1)
        
        for pack in iter:
            print("Get pack", time.time()-t1)
            # get sentence from pack
            sentences = list(pack.get(Sentence))
            num_sent = len(sentences)
            first_sent = sentences[0]
            # delete first sentence
            pack.delete_entry(first_sent)
            self.assertEqual(len(list(pack.get_data(Sentence))), num_sent - 1)
            print("get&delete sentence", time.time()-t1)

    def test_get_raw(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2-t1)
        
        for pack in iter:
            print("Get pack", time.time()-t1)
            # get sentence from pack
            sentences = list(pack.get_raw(Sentence))
            num_sent = len(sentences)
            self.assertNotEqual(num_sent, 0)
            # delete first sentence
            print("get&delete sentence", time.time()-t1)


    def test_get_attributes(self):
        t1 = time.time()
        for pack in self.nlp.process_dataset(self.dataset_path):
            # case 1: get sentence context from the beginning
            requests = {
                Sentence: ["speaker"],
                # Token: ["pos", "sense"],
                # EntityMention: []
            }
            instances = list(
                pack.get_data(Sentence, request=requests, skip_k=1)
            )
            self.assertIsNotNone(instances)
        
            # instances = list(pack.get_data(Sentence))
            # self.assertEqual(len(instances), 2)
            # self.assertEqual(
            #     instances[1]["offset"], len(instances[0]["context"]) + 1
            # )

            # # case 2: get sentence context from the second instance
            # instances = list(pack.get_data(Sentence, skip_k=1))
            # self.assertEqual(len(instances), 1)
            # self.assertEqual(instances[0]["offset"], 165)

            # # case 3: get document context
            # instances = list(pack.get_data(Document, skip_k=0))
            # self.assertEqual(len(instances), 1)
            # self.assertEqual(instances[0]["offset"], 0)

            # # case 3.1: test get single
            # document: Document = pack.get_single(Document)
            # self.assertEqual(document.text, instances[0]["context"])

            # # case 4: test offset out of index
            # instances = list(pack.get_data(Sentence, skip_k=10))
            # self.assertEqual(len(instances), 0)

        print("test_get_attributes", time.time()-t1)


if __name__ == "__main__":
    test = OntonoteGetterPipelineTest()
    test.setUp()
    # test.test_get_raw()
    test.test_get_attributes()
    # unittest.main()