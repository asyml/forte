import logging
import yaml

from texar.torch.hyperparams import HParams
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import (
    CoNLLNEREvaluator, CoNLLNERPredictor)
from forte.train_pipeline import TrainPipeline
from forte.trainer.ner_trainer import CoNLLNERTrainer
from examples.NER.ner_vocab_processor import CoNLL03VocabularyProcessor

logging.basicConfig(level=logging.DEBUG)


def main():
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))

    all_config = {}
    all_config.update(config_data)
    all_config.update(config_model)

    config = HParams(all_config, default_hparams=None)

    reader = CoNLL03Reader(lazy=False)

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
    train_pipe.train()


if __name__ == '__main__':
    main()
