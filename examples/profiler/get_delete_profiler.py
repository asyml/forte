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
import time

from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Document, Token


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
        self.nlp.set_profiling(True)
        self.nlp.initialize()
        self.data_pack: DataPack = self.nlp.process_one(self.dataset_path)

    def test_delete(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        count_del = 0
        c = 0
        for pack in iter:
            print(pack)
            # get sentence from pack
            sentences = list(pack.get(Sentence))
            num_sent = len(sentences)
            first_sent = sentences[0]
            # delete first sentence
            t3 = time.time()
            pack.delete_entry_new(0)
            count_del += time.time() - t3
        print("Delete pack avg: ", count_del / c)

    def test_get_entry_raw(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        count_setence = 0
        count_token = 0
        sent_total = 0
        token_total = 0
        for pack in iter:
            t3 = time.time()
            # get sentence from pack
            for sent in pack.get_raw(Sentence):
                # get tokens from every sentence.
                token_entries = pack.get_raw(
                    entry_type=Token,
                    # range_annotation=sent
                )
                tok = list(token_entries)
                num_tok = len(tok)
                token_total += num_tok
            count_token += time.time() - t3
            self.assertNotEqual(num_tok, 0)

            t4 = time.time()
            sentence = list(pack.get_raw(Sentence))
            count_setence += time.time() - t4
            num_sent = len(sentence)
            sent_total += num_sent
            self.assertNotEqual(num_sent, 0)

        print("Get sentence avg: ", count_setence / sent_total)
        print("Get token avg: ", count_token / token_total)

    def test_get_raw(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        get_total = 0
        pack_total = 0
        for pack in iter:
            t3 = time.time()
            sentences = list(pack.get_raw(Sentence))
            get_total += time.time() - t3
            pack_total += 1
        print("Get pack avg: ", get_total / pack_total)

    def test_get_request(self):
        pack_total = 0
        t1 = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        get_total = 0
        print("process_dataset", t2 - t1)
        for pack in iter:
            pack_total += 1
            # case 1: get sentence context from the beginning
            requests = {
                Sentence: ["speaker"],
                # Token: ["pos", "sense"],
                # EntityMention: []
            }
            t3 = time.time()
            instances = list(
                pack.get_data(Sentence, request=requests, skip_k=1)
            )
            get_total += time.time() - t3
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

        print("Get attribute avg: ", get_total / pack_total)

    def test_get_raw_request(self):
        pack_total = 0
        t1 = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        get_total = 0
        print("process_dataset", t2 - t1)
        for pack in iter:
            pack_total += 1
            # case 1: get sentence context from the beginning
            requests = {
                Sentence: ["speaker"],
                # Token: ["pos", "sense"],
                # EntityMention: []
            }
            t3 = time.time()
            instances = list(
                pack.get_data_raw(Sentence, request=requests, skip_k=1)
            )
            get_total += time.time() - t3
            self.assertIsNotNone(instances)
        print("Get attribute avg: ", get_total / pack_total)

    def test_get_attribute(self):
        pack_total = 0
        t1 = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        get_total = 0
        for pack in iter:
            pack_total += 1
            t3 = time.time()
            pack.get_attributes(0, "pos")
            get_total += time.time() - t3
        print("Get attribute avg: ", get_total / pack_total)

    # for old data pack structure only
    def test_get(self):
        t1 = time.time()
        # get processed pack from dataset
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        pack_total = 0
        get_total = 0
        for pack in iter:
            pack_total += 1
            t3 = time.time()
            sentences = list(pack.get(Sentence))
            get_total += time.time() - t3
        print("Get pack avg: ", get_total / pack_total)

    def test_iter_pack(self):
        pack_total = 0
        t1 = time.time()
        iter = self.nlp.process_dataset(self.dataset_path)
        t2 = time.time()
        print("process_dataset", t2 - t1)
        for pack in iter:
            pack_total += 1
        get_total = time.time() - t2
        print("Get attribute avg: ", get_total / pack_total)


if __name__ == "__main__":
    test = OntonoteGetterPipelineTest()
    test.setUp()
    test.test_get_raw_request()
    # unittest.main()
