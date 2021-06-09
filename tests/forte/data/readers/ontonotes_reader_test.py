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
Unit tests for OntonotesReader.
"""
import os
import unittest
from typing import Tuple, List

from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence, PredicateLink


class DummyPackProcessor(PackProcessor):

    def _process(self, input_pack: DataPack):
        pass


class OntonotesReaderPipelineTest(unittest.TestCase):

    def setUp(self):
        root_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, os.pardir, os.pardir, os.pardir
        ))
        # Define and config the Pipeline
        self.dataset_path = os.path.join(root_path, "data_samples/ontonotes/00")
        self.dataset_path_nested_span = os.path.join(
            root_path, "data_samples/ontonotes/nested_spans")

        self.nlp = Pipeline[DataPack]()

        self.nlp.set_reader(OntonotesReader())
        self.nlp.add(DummyPackProcessor())

        self.nlp.initialize()

    def test_process_next(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get(Sentence):
                doc_exists = True
                sent_text = sentence.text
                # second method to get entry in a sentence
                tokens = [token.text for token in pack.get(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))
        self.assertTrue(doc_exists)

    def test_nested_spans(self):
        expected: List[Tuple] = [
            ("bring", "Tomorrow", "ARG0"),
            ("bring", "Ehud Barak and Yasser Arafat", "ARG2"),
            ("bring", "to the resort city of Sharm El", "ARG4"),
            ("have", "years in terms of their statements and attitudes , "
                     "six years or", "ARG0"),
            ("statements", "their", "ARG0"),
            ("statements", "or more -- before the Oslo accords ,", "ARG1")
        ]

        actual: List[Tuple] = []

        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path_nested_span):
            # get sentence from pack
            for sentence in pack.get(Sentence):
                for pred_link in pack.get(PredicateLink, sentence):
                    pred_link: PredicateLink
                    actual.append((pred_link.get_parent().text,
                                   pred_link.get_child().text,
                                   pred_link.arg_type))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
