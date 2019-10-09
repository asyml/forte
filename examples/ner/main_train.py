import logging
import yaml

from texar.torch.hyperparams import HParams
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import (
    CoNLLNEREvaluator, CoNLLNERPredictor)
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
