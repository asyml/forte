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
This class is used for data selection of the data augmentation tasks.
For each specific dataset, we first create a reader that reads
in the document text as Datapack.
Then we pipeline the reader and the ElasticSearchIndexProcessor
to create a indexer,
which is defined in forte/data/indexers/EleasticSearchIndexer.
This class is then used by the DataSelector class to search for documents.
"""

from typing import Dict, Any

from forte.common.configuration import Config
from forte.data.readers.base_reader import PackReader
from forte.data.data_pack import DataPack
from forte.data.readers import MSMarcoPassageReader
from forte.indexers.elastic_indexer import ElasticSearchIndexer
from forte.pipeline import Pipeline
from forte.processors.data_augment import DataSelectorIndexProcessor


__all__ = [
    "CreateIndexerBasePipeline",
    "CreateIndexerMSMacroPipeline",
]


class CreateIndexerBasePipeline:

    def __init__(self):
        self.reader = self._set_reader()
        self.datapath = self._set_data_path()
        self.config = Config(self.default_config(), default_hparams=None)
        self.create_pipeline()
        self.create_index()

    def _set_reader(self) -> PackReader:
        """
        Return: A dataset Reader.
        """
        raise NotImplementedError

    def _set_data_path(self) -> str:
        """
        Return: The dataset path used in the Reader.
        """
        raise NotImplementedError

    def create_pipeline(self):
        self.nlp: Pipeline[DataPack] = Pipeline()
        self.nlp.set_reader(self.reader)
        self.nlp.add(DataSelectorIndexProcessor(), config=self.config)
        self.nlp.initialize()

    def create_index(self):
        for idx, _ in enumerate(self.nlp.process_dataset(self.datapath)):
            if idx + 1 > 0 and (idx + 1) % 10000 == 0:
                print(f"Indexed {idx + 1} packs")

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


class CreateIndexerMSMacroPipeline(CreateIndexerBasePipeline):
    def __init__(self):
        super().__init__()

    def _set_reader(self) -> PackReader:
        return MSMarcoPassageReader()

    def _set_data_path(self) -> str:
        return 'data_samples/ms_marco_passage_retrieval'

    @classmethod
    def default_configs(cls):
        config: Dict = super().default_configs()
        return config
