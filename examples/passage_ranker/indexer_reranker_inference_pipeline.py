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
from typing import cast
from termcolor import colored
import yaml

import texar.torch as tx

from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackTerminalReader

from forte.pipeline import Pipeline

from forte.processors.ir import (
    ElasticSearchQueryCreator, ElasticSearchProcessor, BertRerankingProcessor)

from ft.onto.base_ontology import Sentence


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(config_file, "r"))
    config = tx.HParams(config, default_hparams=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             config.data.relative_path)

    doc_pack_name = config.indexer.response_pack_name_prefix

    nlp = Pipeline()
    nlp.set_reader(reader=MultiPackTerminalReader(), config=config.reader)

    # Indexing and Re-ranking
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(),
                      config=config.indexer)
    nlp.add_processor(processor=BertRerankingProcessor(),
                      config=config.reranker)

    nlp.initialize()

    passage_keys = [f"passage_{i}" for i in range(config.query_creator.size)]
    num_passages = len(passage_keys)
    print(f"Retrieved {num_passages} passages.")

    for m_pack_ in nlp.process_dataset():
        m_pack = cast(MultiPack, m_pack_)
        for p, passage in enumerate(passage_keys):
            pack = m_pack.get_pack(passage)
            print(colored(f"Passage: #{p}", "green"), pack.text, "\n")
            for s, sentence in enumerate(pack.get(Sentence)):
                sent_text = sentence.text
                print(colored(f"Sentence #{s}:", 'green'), sent_text, "\n")
            if p < num_passages:
                input(colored("Press ENTER to get next result...\n", 'green'))

    print(colored('#' * 20, 'blue'))
