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

import logging

import yaml

from ner_vocab_processor import CoNLL03VocabularyProcessor
from forte.common.configuration import Config
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from forte.processors.nlp import CoNLLNERPredictor
from forte.train_pipeline import TrainPipeline
from forte.trainer.ner_trainer import CoNLLNERTrainer

logging.basicConfig(level=logging.DEBUG)


def main():
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    config_evaluator = yaml.safe_load(open("config_evaluator.yml", "r"))
    config_preprocess = yaml.safe_load(open("config_preprocessor.yml", "r"))

    # All the configs
    config = Config({}, default_hparams=None)
    config.add_hparam("config_data", config_data)
    config.add_hparam("config_model", config_model)
    config.add_hparam("preprocessor", config_preprocess)
    config.add_hparam("reader", {})
    config.add_hparam("evaluator", config_evaluator)

    reader = CoNLL03Reader()

    # Keep the vocabulary processor as a simple counter
    vocab_processor = CoNLL03VocabularyProcessor()

    ner_trainer = CoNLLNERTrainer()
    ner_predictor = CoNLLNERPredictor()
    ner_evaluator = CoNLLNEREvaluator()

    train_pipe = TrainPipeline(
        train_reader=reader,
        trainer=ner_trainer,
        dev_reader=reader,
        configs=config,
        preprocessors=[vocab_processor],
        predictor=ner_predictor,
        evaluator=ner_evaluator,
    )
    train_pipe.run()


if __name__ == "__main__":
    main()
