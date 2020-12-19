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

import os

from examples.data_augmentation.data_select import CreateIndexerPipeline
from forte.data.readers import MSMarcoPassageReader


def main():
    indexer_name = "da_selector"

    file_dir_path = os.path.dirname(__file__)
    data_dir = 'data_samples/ms_marco_passage_retrieval'
    abs_data_dir = os.path.abspath(
        os.path.join(file_dir_path, *([os.pardir] * 3), data_dir))

    indexer_config = {
        "batch_size": 5,
        "fields":
            ["doc_id", "content", "pack_info"],
        "indexer": {
            "name": "ElasticSearchIndexer",
            "hparams":
                {"index_name": indexer_name,
                 "hosts": "localhost:9200",
                 "algorithm": "bm25"},
            "other_kwargs": {
                "request_timeout": 10,
                "refresh": False
            }
        }
    }

    reader = MSMarcoPassageReader()
    nlp = CreateIndexerPipeline(reader=reader, reader_config=None,
                                indexer_config=indexer_config)
    nlp.create_index(abs_data_dir)


if __name__ == "__main__":
    main()
