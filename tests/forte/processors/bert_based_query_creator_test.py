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
Unit tests for Query Creator processor.
"""
import unittest
import os
import tempfile
import shutil

from ddt import ddt, data, unpack

from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.ir import BertBasedQueryCreator
from forte.data.readers import MultiPackSentenceReader
from forte.data.ontology import Query


@ddt
class TestBertBasedQueryCreator(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @data((["Hello, good morning",
            "This is a tool for NLP"],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx+1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline[MultiPack]()
        reader_config = {"input_pack_name": "query",
                         "output_pack_name": "output"}
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        config = {"model": {"name": "bert-base-uncased"},
                  "tokenizer": {"name": "bert-base-uncased"},
                  "max_seq_length": 128,
                  "query_pack_name": "query"}
        nlp.add(BertBasedQueryCreator(), config=config)

        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            query_pack = m_pack.get_pack("query")
            self.assertEqual(len(query_pack.generics), 1)
            self.assertIsInstance(query_pack.generics[0], Query)
            query = query_pack.generics[0].value
            self.assertEqual(query.shape, (1, 768))
