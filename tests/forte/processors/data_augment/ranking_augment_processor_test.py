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
Unit tests for query ranking augment processors
"""

import unittest
import os
import tempfile
import shutil

from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackSentenceReader
from forte.processors.ir import BertBasedQueryCreator
from forte.processors.data_augment.query_ranking_augment_processor import RankingDataAugmentProcessor

from ddt import ddt, data, unpack

@ddt
class TestRankingAugmentProcessor(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)

        config = {"model": {"name": "bert-base-uncased"},
                  "tokenizer": {"name": "bert-base-uncased"},
                  "max_seq_length": 128,
                  "query_pack_name": "query"}
        nlp.add(BertBasedQueryCreator(), config=config)

        data_augment_config = {
            'augment_algorithm': "DictionaryReplacement",
            'augment_ontologies': ["Sentence"],
            'replacement_prob': 0.9,
            'replacement_level': 'word',
            'query_pack_name': 'query',
            'aug_query': 'true',
            'aug_document': 'false',
            'kwargs': {
                "lang": "eng"
            }
        }
        nlp.add(RankingDataAugmentProcessor(),
                config=data_augment_config)

        nlp.initialize()

        expected_outputs = [
            "Blessed Virgin and Samantha go far at the motorbus station early on but wait until twelve noon for the bus topology."
        ]

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            query_pack = m_pack.get_pack("query")
            aug_query_pack = m_pack.get_pack("aug_query")
            self.assertEqual(aug_query_pack.text, expected_outputs[idx])


if __name__ == "__main__":
    unittest.main()
