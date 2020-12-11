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
Unit tests for CoNLL03Reader.
"""

import unittest

from forte.data.data_pack import DataPack
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence, EntityMention


class DummyPackProcessor(PackProcessor):

    def _process(self, input_pack: DataPack):
        pass


class CoNLL03ReaderPipelineTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03_new"
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(CoNLL03Reader())
        self.nlp.initialize()

    def test_process_next(self):
        doc_exists = False

        expected_sentence = ['The', 'European', 'Commission', 'said', 'on',
            'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to',
            'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists',
            'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be',
            'transmitted', 'to', 'sheep', '.']
        expected_ner_type = ["ORG", "MISC", "MISC"]
        expected_token = ["European Commission", "German", "British"]

        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get(Sentence):
                doc_exists = True
                # sent_text sentence.text
                # second method to get entry in a sentence
                tokens = [token.text for token in pack.get(Token, sentence)]
                print(tokens)
                self.assertEqual(expected_sentence, tokens)

                i = 0
                for ner in pack.get(EntityMention, sentence):
                    print(ner.text)
                    print(ner.ner_type)
                    self.assertEqual(expected_token[i], ner.text)
                    self.assertEqual(expected_ner_type[i], ner.ner_type)
                    i += 1

        self.assertTrue(doc_exists)


if __name__ == '__main__':
    unittest.main()
