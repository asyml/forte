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
Unit tests for Data Selector.
"""
import unittest
import os
import tempfile

from ddt import ddt, data, unpack

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.data_augment.selector_index_processor \
    import DataSelectorIndexProcessor
from forte.data.readers import MSMarcoPassageReader
from forte.indexers.elastic_indexer import ElasticSearchIndexer
from forte.processors.base.data_selector_for_da import \
    QueryDataSelector, RandomDataSelector


@ddt
class TestDataSelector(unittest.TestCase):

    def setUp(self):
        # create indexer
        file_dir_path = os.path.dirname(__file__)
        data_dir = 'data_samples/ms_marco_passage_retrieval'
        self.abs_data_dir = os.path.abspath(os.path.join(file_dir_path,
                                                         *([os.pardir] * 4),
                                                         data_dir))
        self.index_name = "final"
        indexer_config = {
            "batch_size": 5,
            "fields":
                ["doc_id", "content", "pack_info"],
            "indexer": {
                "name": "ElasticSearchIndexer",
                "hparams":
                    {"index_name": self.index_name,
                     "hosts": "localhost:9200",
                     "algorithm": "bm25"},
                "other_kwargs": {
                    "request_timeout": 10,
                    "refresh": True
                }
            }
        }
        self.indexer = ElasticSearchIndexer(
            config={"index_name": self.index_name})
        nlp: Pipeline[DataPack] = Pipeline()
        nlp.set_reader(MSMarcoPassageReader())
        nlp.add(DataSelectorIndexProcessor(), config=indexer_config)
        nlp.initialize()

        self.size = 0
        for _ in nlp.process_dataset(self.abs_data_dir):
            self.size += 1

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.indexer.elasticsearch.indices.delete(
            index=self.index_name, ignore=[400, 404])

    @data((["scientific minds"]))
    @unpack
    def test_query_data_selector(self, text):
        file_path = os.path.join(self.test_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write(text)

        config = {"index_config":
                      {"index_name": self.index_name},
                  "size": 3,
                  "field": "content"}
        nlp: Pipeline[DataPack] = Pipeline()
        nlp.set_reader(QueryDataSelector(), config=config)
        nlp.initialize()

        text = []
        for pack in nlp.process_dataset(file_path):
            text.append(pack.text)

        self.assertEqual(len(text), 1)
        self.assertIn("The presence of communication amid scientific minds was",
                      text[0])

    def test_random_data_selector(self):
        size = 2
        config = {"index_config":
                      {"index_name": self.index_name},
                  "size": size}
        nlp: Pipeline[DataPack] = Pipeline()
        nlp.set_reader(RandomDataSelector(), config=config)
        nlp.initialize()

        text = []
        for pack in nlp.process_dataset():
            text.append(pack.text)

        self.assertEqual(len(text), size)


if __name__ == "__main__":
    unittest.main()
