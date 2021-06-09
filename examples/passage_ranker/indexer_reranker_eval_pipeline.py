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

import argparse

import yaml

from forte.elastic import ElasticSearchQueryCreator, \
    ElasticSearchProcessor

from ms_marco_evaluator import MSMarcoEvaluator
from reader import EvalReader
from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.ir import BertRerankingProcessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="./config.yml",
                        help="Config YAML filepath")
    args = parser.parse_args()

    # loading config
    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    # reading query input file
    parser.add_argument("--input_file",
                        default="./data/collectionandqueries/query_doc_id.tsv",
                        help="Input query filepath")

    input_file = config.evaluator.input_file

    # initializing pipeline with processors
    nlp: Pipeline = Pipeline[MultiPack]()
    eval_reader = EvalReader()
    nlp.set_reader(reader=eval_reader, config=config.reader)
    nlp.add(ElasticSearchQueryCreator(), config=config.query_creator)
    nlp.add(ElasticSearchProcessor(), config=config.indexer)
    nlp.add(BertRerankingProcessor(), config=config.reranker)
    nlp.add(MSMarcoEvaluator(), config=config.evaluator)
    nlp.initialize()

    for idx, m_pack in enumerate(nlp.process_dataset(input_file)):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} examples")

    scores = nlp.evaluate()
    print(scores)
