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
Unit tests for Data Selector Index Processor.
"""
import unittest
import os

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.data_augment.selector_index_processor \
    import DataSelectorIndexProcessor
from forte.data.readers import MSMarcoPassageReader
from forte.indexers.elastic_indexer import ElasticSearchIndexer


class TestDataSelectorIndexProcessor(unittest.TestCase):

    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        data_dir = 'data_samples/ms_marco_passage_retrieval'
        self.abs_data_dir = os.path.abspath(os.path.join(file_dir_path,
                                                         *([os.pardir] * 4),
                                                         data_dir))
        corpus_file = os.path.join(self.abs_data_dir, 'collection.tsv')

        self.expected_content = set()
        with open(corpus_file, 'r') as f:
            for line in f.readlines():
                key, value = tuple(line.split('\t', 1))
                self.expected_content.add(value)

        self.index_name = "test_indexer"
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

        self.nlp: Pipeline[DataPack] = Pipeline()
        self.reader = MSMarcoPassageReader()
        self.processor = DataSelectorIndexProcessor()
        self.nlp.set_reader(self.reader)
        self.nlp.add(self.processor, config=indexer_config)
        self.nlp.initialize()

    def tearDown(self):
        self.indexer.elasticsearch.indices.delete(
            index=self.index_name, ignore=[400, 404])

    def test_pipeline(self):
        size = 0
        for _ in self.nlp.process_dataset(self.abs_data_dir):
            size += 1

        retrieved_document = self.indexer.search(
            query={"query": {"match_all": {}}}, index_name=self.index_name,
            size=size)

        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), size)
        results = set([hit["_source"]["content"] for hit in hits])
        self.assertEqual(results, self.expected_content)
