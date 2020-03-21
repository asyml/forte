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

import os
import yaml

import texar.torch as tx

from forte.pipeline import Pipeline
from forte.processors.ir import (
    ElasticSearchQueryCreator, ElasticSearchProcessor, BertRerankingProcessor)

from examples.passage_reranker.reader import EvalReader
from examples.passage_reranker.ms_marco_evaluator import MSMarcoEvaluator


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(config_file, "r"))
    config = tx.HParams(config, default_hparams=None)
    ms_marco_evaluator = MSMarcoEvaluator()

    nlp = Pipeline()
    eval_reader = EvalReader()
    nlp.set_reader(reader=eval_reader, config=config.reader)
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(),
                      config=config.indexer)
    nlp.add_processor(processor=BertRerankingProcessor(),
                      config=config.reranker)

    nlp.set_evaluator(evaluator=ms_marco_evaluator,
                      config=config.evaluator)
    nlp.initialize()

    file_path = os.path.join(os.path.dirname(__file__),
                             "data/collectionandqueries/queries.dev.small.tsv")
    for idx, m_pack in enumerate(nlp.process_dataset(file_path)):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} examples")

    scores = nlp.evaluate()
    print(scores)
