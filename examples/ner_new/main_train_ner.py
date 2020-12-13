#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
from typing import Iterator, Dict
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from texar.torch.data import Batch
from tqdm import tqdm
import yaml

from examples.ner_new.ner_evaluator import CoNLLNEREvaluator
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.pipeline import Pipeline
from forte.predictor import Predictor
from forte.common.configuration import Config
from forte.data.extractor import AttributeExtractor
from forte.data.extractor import \
    BaseExtractor, BioSeqTaggingExtractor, CharExtractor
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from ft.onto.base_ontology import Sentence, Token, EntityMention

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def create_model(schemes: Dict[str, Dict[str, BaseExtractor]],
                 config: Config):
    text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
    char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
    ner_extractor: BaseExtractor = schemes["ner_tag"]["extractor"]

    _model: nn.Module = \
        BiRecurrentConvCRF(word_vocab=text_extractor.get_dict(),
                           char_vocab_size=char_extractor.size(),
                           tag_vocab_size=ner_extractor.size(),
                           config_model=config)

    return _model


def train(_model: BiRecurrentConvCRF, _optim: Optimizer, _batch: Batch):
    word = _batch["text_tag"]["data"]
    char = _batch["char_tag"]["data"]
    ner = _batch["ner_tag"]["data"]
    word_masks = _batch["text_tag"]["masks"][0]

    _optim.zero_grad()

    loss = _model(word, char, ner, mask=word_masks)

    loss.backward()
    _optim.step()

    _batch_train_err = loss.item() * _batch.batch_size

    return _batch_train_err


def predict_forward_fn(_model: BiRecurrentConvCRF, _batch: Dict) -> Dict:
    word = _batch["text_tag"]["data"]
    char = _batch["char_tag"]["data"]
    word_masks = _batch["text_tag"]["masks"][0]

    output = _model.decode(input_word=word,
                           input_char=char,
                           mask=word_masks)
    output = output.numpy()
    return {'ner_tag': output}


if __name__ == "__main__":
    # All the configs
    config_data = Config({}, default_hparams=yaml.safe_load(
        open("configs/config_data.yml", "r")))
    config_model = Config({}, default_hparams=yaml.safe_load(
        open("configs/config_model.yml", "r")))

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    tp_request: Dict = {
        "scope": Sentence,
        "schemes": {
            "text_tag": {
                "entry_type": Token,
                "vocab_method": "indexing",
                "attribute_get": "text",
                "type": TrainPreprocessor.DATA_INPUT,
                "extractor": AttributeExtractor
            },
            "char_tag": {
                "entry_type": Token,
                "vocab_method": "indexing",
                "max_char_length": config_data.max_char_length,
                "type": TrainPreprocessor.DATA_INPUT,
                "extractor": CharExtractor
            },
            "ner_tag": {
                "entry_type": EntityMention,
                "attribute": "ner_type",
                "based_on": Token,
                "vocab_method": "indexing",
                "type": TrainPreprocessor.DATA_OUTPUT,
                "extractor": BioSeqTaggingExtractor
            }
        }
    }

    # All not specified dataset parameters are set by default in Texar.
    # Default settings can be found here:
    # https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.DatasetBase.default_hparams
    tp_config: Dict = {
        "preprocess": {
            "pack_dir": config_data.train_path,
            "device": device.type
        },
        "dataset": {
            "batch_size": config_data.batch_size_tokens
        }
    }

    ner_train_reader = CoNLL03Reader(cache_in_memory=True)
    ner_val_reader = CoNLL03Reader(cache_in_memory=True)

    train_preprocessor = TrainPreprocessor(train_reader=ner_train_reader,
                                           request=tp_request,
                                           config=tp_config)

    model: BiRecurrentConvCRF = \
        create_model(schemes=train_preprocessor.feature_resource["schemes"],
                     config=config_model)
    model.to(device)

    optim: Optimizer = SGD(model.parameters(),
                           lr=config_model.learning_rate,
                           momentum=config_model.momentum,
                           nesterov=True)

    predictor = Predictor(
        batch_size=train_preprocessor.config.dataset.batch_size,
        model=model,
        predict_forward_fn=predict_forward_fn,
        feature_resource=train_preprocessor.feature_resource,
        cross_pack=False)
    evaluator = CoNLLNEREvaluator()

    val_pl: Pipeline = Pipeline()
    val_pl.set_reader(ner_val_reader)
    val_pl.add(predictor)
    val_pl.add(evaluator)

    epoch = 0
    train_err: int = 0
    train_total: float = 0.0
    train_sentence_len_sum: float = 0.0
    output_file = "tmp_eval.txt"
    score_file = "tmp_eval.score"
    scores: Dict[str, float] = {}

    logger.info("Start training.")
    while epoch < config_data.num_epochs:
        epoch += 1

        # Get iterator of preprocessed batch of train data
        train_batch_iter: Iterator[Batch] = \
            train_preprocessor.get_train_batch_iterator()

        for batch in tqdm(train_batch_iter):
            batch_train_err = train(model, optim, batch)

            train_err += batch_train_err
            train_total += batch.batch_size
            train_sentence_len_sum += \
                torch.sum(batch["text_tag"]["masks"][0]).item()

        logger.info("%dth Epoch training, "
                    "total number of examples: %d, "
                    "Average sentence length: %0.3f, "
                    "loss: %0.3f",
                    epoch, train_total,
                    train_sentence_len_sum / train_total,
                    train_err / train_total)

        train_err = 0
        train_total = 0.0
        train_sentence_len_sum = 0.0

        val_pl.run(config_data.val_path)

        logger.info("%dth Epoch evaluating, "
                    "val result: %s",
                    epoch, evaluator.get_result())

    # Save training state to disk
    # train_preprocessor.save_state(config_data.train_state_path)
    # torch.save(model, config_model.model_path)
