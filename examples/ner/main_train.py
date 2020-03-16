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

from texar.torch.hyperparams import HParams
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import CoNLLNERPredictor
from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from forte.train_pipeline import TrainPipeline
from forte.trainer.ner_trainer import CoNLLNERTrainer
from examples.ner.ner_vocab_processor import CoNLL03VocabularyProcessor

logging.basicConfig(level=logging.DEBUG)


def main():
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    config_preprocess = yaml.safe_load(open("config_preprocessor.yml", "r"))

    config = HParams({}, default_hparams=None)
    config.add_hparam('config_data', config_data)
    config.add_hparam('config_model', config_model)
    config.add_hparam('preprocessor', config_preprocess)

    reader = CoNLL03Reader()

    # Keep the vocabulary processor as a simple counter
    vocab_processor = CoNLL03VocabularyProcessor()

    ner_trainer = CoNLLNERTrainer()
    ner_predictor = CoNLLNERPredictor()
    ner_evaluator = CoNLLNEREvaluator()

    train_pipe = TrainPipeline(train_reader=reader, trainer=ner_trainer,
                               dev_reader=reader, configs=config,
                               preprocessors=[vocab_processor],
                               predictor=ner_predictor, evaluator=ner_evaluator)
    train_pipe.run()


if __name__ == '__main__':
    main()
