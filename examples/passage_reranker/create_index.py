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

import texar.torch as tx

from forte.data.readers import CorpusReader
from forte.processors import ElasticSearchIndexProcessor
from forte.pipeline import Pipeline

if __name__ == "__main__":
    nlp = Pipeline()
    nlp.set_reader(CorpusReader())
    config = tx.HParams({
        "batch_size": 100000,
        "fields": ["doc_id", "content"],
        "indexer": {
            "name": "ElasticSearchIndexer",
            "hparams": {
                "index_name": "elastic_indexer2",
                "hosts": "localhost:9200",
                "algorithm": "bm25"
            },
            "other_kwargs": {
                "request_timeout": 60,
                "refresh": False
            }
        }
    }, default_hparams=None)
    nlp.add_processor(ElasticSearchIndexProcessor(), config=config)
    nlp.initialize()

    for idx, pack in enumerate(nlp.process_dataset(".")):
        if idx + 1 > 0 and (idx + 1) % 100000 == 0:
            print(f"Completed {idx+1} packs")
