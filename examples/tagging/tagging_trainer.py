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

from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from forte.common.configuration import Config
from forte.data import BaseExtractor
from forte.data.data_pack import DataPack
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.pipeline import Pipeline
from forte.processors.base import Predictor
from forte.trainer.base.trainer import BaseTrainer
from forte.utils import create_import_error_msg

logger = logging.getLogger(__name__)


class TaggingTrainer(BaseTrainer):
    def __init__(
        self,
        task_type: str,
        config_data: Config,
        config_model: Config,
        config_extractors: Dict,
        device,
    ):
        super().__init__()
        self.task_type = task_type

        # All the configs
        self.config_data: Config = config_data
        self.config_model: Config = config_model
        self.config_extractors: Dict = config_extractors
        self.device = device

        self.model = None

    def create_tp_config(self) -> Dict:
        return {
            "preprocess": {"device": self.device.type},
            "dataset": {"batch_size": self.config_data.batch_size_tokens},
            "request": self.config_extractors,
        }

    def create_pack_iterator(self) -> Iterator[DataPack]:
        reader = CoNLL03Reader()
        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(reader)
        train_pl.initialize()
        yield from train_pl.process_dataset(self.config_data.train_path)

    def train(self):
        logging.info("Constructing the extractors and models.")
        schemes: Dict = self.train_preprocessor.request["schemes"]
        text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
        char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
        output_extractor: BaseExtractor = schemes["output_tag"]["extractor"]

        self.model: BiRecurrentConvCRF = BiRecurrentConvCRF(
            word_vocab=text_extractor.vocab.to_dict(),
            char_vocab_size=len(char_extractor.vocab),
            tag_vocab_size=len(output_extractor.vocab),
            config_model=self.config_model,
        )
        self.model.to(self.device)

        logging.info("Constructing the optimizer.")
        optim: Optimizer = SGD(
            self.model.parameters(),
            lr=self.config_model.learning_rate,
            momentum=self.config_model.momentum,
            nesterov=True,
        )

        tp = self.train_preprocessor

        logging.info("Constructing the validation pipeline.")
        predictor = TaggingPredictor()
        # Load the extractors to the predictor.
        predictor.set_feature_requests(self.train_preprocessor.request)
        predictor.load(self.model)

        evaluator = CoNLLNEREvaluator()
        output_extractor_configs = self.config_extractors["feature_scheme"][
            "output_tag"
        ]["extractor"]["config"]
        evaluator_config = {
            "entry_type": output_extractor_configs["entry_type"],
            "tagging_unit": output_extractor_configs["tagging_unit"],
            "attribute": output_extractor_configs["attribute"],
        }

        val_reader = CoNLL03Reader(cache_in_memory=True)
        val_pl: Pipeline = Pipeline()
        val_pl.set_reader(val_reader)
        val_pl.add(
            predictor,
            config={
                "batcher": {
                    "batch_size": 10,
                }
            },
        )
        val_pl.add(evaluator, config=evaluator_config)
        val_pl.initialize()

        epoch: int = 0
        train_err: int = 0
        train_total: float = 0.0
        train_sentence_len_sum: float = 0.0

        logger.info("Start training.")
        try:
            from texar.torch.data import (
                Batch,
            )  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                create_import_error_msg(
                    "texar-pytorch", "extractor", "the extractor system"
                )
            ) from e
        while epoch < self.config_data.num_epochs:
            epoch += 1

            # Get iterator of preprocessed batch of train data
            batch_iter: Iterator[Batch] = tp.get_train_batch_iterator()

            for batch in tqdm(batch_iter):
                word = batch["text_tag"]["data"]
                char = batch["char_tag"]["data"]
                output = batch["output_tag"]["data"]
                word_masks = batch["text_tag"]["masks"][0]

                optim.zero_grad()

                loss = self.model(word, char, output, mask=word_masks)

                loss.backward()
                optim.step()

                batch_train_err = loss.item() * batch.batch_size

                train_err += batch_train_err
                train_total += batch.batch_size
                train_sentence_len_sum += torch.sum(
                    batch["text_tag"]["masks"][0]
                ).item()

            logger.info(
                "%dth Epoch training, "
                "total number of examples: %d, "
                "Average sentence length: %0.3f, "
                "loss: %0.3f",
                epoch,
                train_total,
                train_sentence_len_sum / train_total,
                train_err / train_total,
            )

            train_err = 0
            train_total = 0.0
            train_sentence_len_sum = 0.0

            val_pl.run(self.config_data.val_path)

            logger.info(
                "%dth Epoch evaluating, " "val result: %s",
                epoch,
                evaluator.get_result(),
            )


class TaggingPredictor(Predictor):
    def predict(self, data_batch: Dict) -> Dict:
        val_output = self.model.decode(
            input_word=data_batch["text_tag"]["data"],
            input_char=data_batch["char_tag"]["data"],
            mask=data_batch["text_tag"]["masks"][0],
        )
        val_output = val_output.numpy()
        return {"output_tag": val_output}
