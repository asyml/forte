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
import sys
from typing import Iterator, Dict, List
import torch
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from texar.torch.data import Batch
from tqdm import tqdm
import yaml

from examples.tagging.evaluator import CoNLLNEREvaluator
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.pipeline import Pipeline
from forte.predictor import Predictor
from forte.common.configuration import Config
from forte.data.extractor import AttributeExtractor
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.extractor.char_extractor import CharExtractor
from forte.data.extractor.seqtagging_extractor import BioSeqTaggingExtractor
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from ft.onto.base_ontology import Sentence, Token, EntityMention

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def predict_forward_fn(_model: BiRecurrentConvCRF, _batch: Dict) -> Dict:
    val_output = _model.decode(input_word=_batch["text_tag"]["data"],
                               input_char=_batch["char_tag"]["data"],
                               mask=_batch["text_tag"]["masks"][0])
    val_output = val_output.numpy()
    return {'output_tag': val_output}


if __name__ == "__main__":
    task = sys.argv[1]
    assert task in ["ner", "pos"], \
        "Not supported nlp task type: {}".format(task)

    # All the configs
    config_data = Config({}, default_hparams=yaml.safe_load(
        open("configs/config_data.yml", "r")))
    config_model = Config({}, default_hparams=yaml.safe_load(
        open("configs/config_model.yml", "r")))

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    # Generate request
    text_extractor: AttributeExtractor = \
        AttributeExtractor(config={"entry_type": Token,
                                   "vocab_method": "indexing",
                                   "attribute_get": "text"})

    char_extractor: CharExtractor = \
        CharExtractor(config={"entry_type": Token,
                              "vocab_method": "indexing",
                              "max_char_length": config_data.max_char_length})

    # Add output part in request based on different task type
    output_extractor: BaseExtractor
    if task == "ner":
        output_extractor = \
            BioSeqTaggingExtractor(config={"entry_type": EntityMention,
                                           "attribute": "ner_type",
                                           "based_on": Token,
                                           "vocab_method": "indexing"})

    else:
        output_extractor = \
            AttributeExtractor(config={"entry_type": Token,
                                       "attribute_get": "pos",
                                       "vocab_method": "indexing"})

    tp_request: Dict = {
        "scope": Sentence,
        "schemes": {
            "text_tag": {
                "type": TrainPreprocessor.DATA_INPUT,
                "extractor": text_extractor
            },
            "char_tag": {
                "type": TrainPreprocessor.DATA_INPUT,
                "extractor": char_extractor
            },
            "output_tag": {
                "type": TrainPreprocessor.DATA_OUTPUT,
                "extractor": output_extractor
            }
        }
    }

    # All not specified dataset parameters are set by default in Texar.
    # Default settings can be found here:
    # https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.DatasetBase.default_hparams
    tp_config: Dict = {
        "preprocess": {
            "device": device.type
        },
        "dataset": {
            "batch_size": config_data.batch_size_tokens
        }
    }

    # Create data pack generator from reader
    train_reader = CoNLL03Reader(cache_in_memory=True)
    train_pl: Pipeline = Pipeline()
    train_pl.set_reader(train_reader)
    train_pl.initialize()
    train_pack_generator = train_pl.process_dataset(config_data.train_path)

    # Create train preprocessor
    train_preprocessor = TrainPreprocessor(pack_generator=train_pack_generator,
                                           request=tp_request,
                                           config=tp_config)

    model: BiRecurrentConvCRF = \
        BiRecurrentConvCRF(word_vocab=text_extractor.get_dict(),
                           char_vocab_size=char_extractor.size(),
                           tag_vocab_size=output_extractor.size(),
                           config_model=config_model)
    model.to(device)

    optim: Optimizer = SGD(model.parameters(),
                           lr=config_model.learning_rate,
                           momentum=config_model.momentum,
                           nesterov=True)

    predictor = Predictor(
        batch_size=train_preprocessor.config.dataset.batch_size,
        model=model,
        predict_forward_fn=predict_forward_fn,
        feature_resource=train_preprocessor.request,
        cross_pack=False)
    evaluator = CoNLLNEREvaluator()

    val_reader = CoNLL03Reader(cache_in_memory=True)
    val_pl: Pipeline = Pipeline()
    val_pl.set_reader(val_reader)
    val_pl.add(predictor)
    val_pl.add(evaluator)

    epoch = 0
    train_err: int = 0
    train_total: float = 0.0
    train_sentence_len_sum: float = 0.0
    val_scores: List = []
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
            word = batch["text_tag"]["data"]
            char = batch["char_tag"]["data"]
            output = batch["output_tag"]["data"]
            word_masks = batch["text_tag"]["masks"][0]

            optim.zero_grad()

            loss = model(word, char, output, mask=word_masks)

            loss.backward()
            optim.step()

            batch_train_err = loss.item() * batch.batch_size

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
