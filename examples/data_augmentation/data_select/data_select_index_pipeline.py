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
# pylint: disable=useless-super-delegation

"""
This class is used for data selection step in data augmentation tasks.
For each specific dataset, we first create a reader that reads
in the document text as Datapack.
Then we pipeline the reader and the ElasticSearchIndexProcessor
to create an indexer,
which is defined in forte/data/indexers/EleasticSearchIndexer.
This indexer is then used by the DataSelector class to search for documents.
"""

from typing import Dict, Any
import logging

from forte.elastic import ElasticSearchIndexer
from forte.elastic import ElasticSearchPackIndexProcessor

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

__all__ = [
    "CreateIndexerPipeline",
]


class CreateIndexerPipeline:

    def __init__(self, reader, reader_config, indexer_config=None):
        self.reader = reader
        self.reader_config = reader_config
        self.config = indexer_config if indexer_config is not None \
            else self.default_config()
        self.config = Config(self.config, default_hparams=None)
        self.create_pipeline()

    def create_pipeline(self):
        # pylint: disable=attribute-defined-outside-init
        self.nlp: Pipeline[DataPack] = Pipeline()
        self.nlp.set_reader(reader=self.reader, config=self.reader_config)
        self.nlp.add(ElasticSearchPackIndexProcessor(), config=self.config)
        self.nlp.initialize()

    def create_index(self, datapath):
        for idx, _ in enumerate(self.nlp.process_dataset(datapath)):
            if idx + 1 > 0 and (idx + 1) % 10000 == 0:
                logging.info("Indexed %d packs", idx + 1)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "batch_size": 10000,
            "fields": ["doc_id", "content", "pack_info"],
            "indexer": {
                "name": "ElasticSearchIndexer",
                "hparams": ElasticSearchIndexer.default_configs(),
                "other_kwargs": {
                    "request_timeout": 60,
                    "refresh": False
                }
            }
        }
