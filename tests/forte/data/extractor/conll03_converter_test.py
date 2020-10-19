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
Unit test for Conll03_Converter.
"""

import unittest

from forte.data.data_pack import DataPack
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.converter.conll03_converter import CoNLL03Converter
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence


class CoNLL03ReaderPipelineTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03"

    def test_converter(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        converter = CoNLL03Converter()

        # Test token2id and labelpos2id dict
        tokens = ['U.N.', 'official', 'Ekeus', 'heads', 'for', 'Baghdad', '.']
        ners = ['I-ORG', 'O', 'I-PER', 'O', 'O', 'I-LOC', 'O']

        for t in tokens:
            self.assertIn(t, converter.token2id)

        self.assertSetEqual(converter.positions, set(['B', 'I']))
        for ner in ners:
            for pos in converter.positions:
                self.assertIn((ner, pos), converter.labelpos2id)

        self.assertEqual(len(converter.token2id.keys()),
                            len(converter.token2id.values()))
        self.assertEqual(converter.labelpos2id[('O', 'B')],
                            converter.labelpos2id[('O', 'I')])

        expected_token_ids = [converter.token2id[t] for t in tokens]
        expected_ner_ids = [
            converter.labelpos2id[('I-ORG', 'B')],
            converter.labelpos2id[('O', 'B')],
            converter.labelpos2id[('I-PER', 'B')],
            converter.labelpos2id[('O', 'B')],
            converter.labelpos2id[('O', 'I')],
            converter.labelpos2id[('I-LOC', 'B')],
            converter.labelpos2id[('O', 'B')]
        ]


        # get processed pack from dataset
        for pack in pipeline.process_dataset(self.dataset_path):
            for token_ids, ner_ids in converter.convert(pack):
                self.assertEqual(token_ids, expected_token_ids)
                self.assertEqual(ner_ids, expected_ner_ids)

if __name__ == '__main__':
    unittest.main()
