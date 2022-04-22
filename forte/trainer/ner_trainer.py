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

""" This module contains an implementation of the NER trainer.
"""

# pylint: disable=logging-fstring-interpolation
import logging
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Dict

import numpy as np

try:
    import torch
    from torch.optim import SGD
    import torchtext
except ImportError as e:
    raise ImportError(
        " `pytorch` is not installed correctly."
        " Please refer to [extra requirement for models](pip install forte[models])"
        " for more information. "
    ) from e
from tqdm import tqdm

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.models.ner import utils
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.trainer.base.base_trainer import BaseTrainer
from ft.onto.base_ontology import Token, Sentence

logger = logging.getLogger(__name__)


class CoNLLNERTrainer(BaseTrainer):
    def __init__(self):
        """Create an NER trainer."""

        super().__init__()

        self.model = None

        self.word_alphabet = None
        self.char_alphabet = None
        self.ner_alphabet = None

        self.config_model = None
        self.config_data = None
        self.normalize_func = None

        self.device = None
        self.optim, self.trained_epochs = None, None

        self.resource: Optional[Resources] = None

        self.train_instances_cache = []

        # Just for recording
        self.max_char_length = 0

        self.__past_dev_result = None

    def initialize(self, resources: Resources, configs: Config):
        """
        The training pipeline will run this initialization method during
        the initialization phase and send resources in as parameters.

        Args:
            resources: The resources shared in the pipeline.
            configs: configuration object for this trainer.

        Returns:

        """
        self.resource = resources

        self.word_alphabet = resources.get("word_alphabet")
        self.char_alphabet = resources.get("char_alphabet")
        self.ner_alphabet = resources.get("ner_alphabet")

        word_embedding_table = resources.get("word_embedding_table")

        self.config_model = configs.config_model
        self.config_data = configs.config_data

        self.normalize_func = utils.normalize_digit_word

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        utils.set_random_seed(self.config_model.random_seed)

        self.model = BiRecurrentConvCRF(
            word_embedding_table,
            self.char_alphabet.size(),
            self.ner_alphabet.size(),
            self.config_model,
        ).to(device=self.device)

        self.optim = SGD(
            self.model.parameters(),
            lr=self.config_model.learning_rate,
            momentum=self.config_model.momentum,
            nesterov=True,
        )

        self.trained_epochs = 0

        self.resource.update(model=self.model)

    def data_request(self) -> Dict:
        """
        Build a request to a :class:`DataPack <forte.data.data_pack.DataPack>`.
        The NER trainer will request the ner filed of the tokens, organized
        by each sentence.

        Returns:

        """

        request_string = {
            "context_type": Sentence,
            "request": {
                Token: ["ner"],
                Sentence: [],  # span by default
            },
        }
        return request_string

    def consume(self, instance):
        """
        Consumes the next NER instance for training. The instance will be
        collected until a batch is full.

        Args:
            instance: An instance contains a sentence, in the form of tokens and
                their "ner" tag.

        Returns:

        """

        tokens = instance["Token"]
        word_ids = []
        char_id_seqs = []
        ner_tags, ner_ids = tokens["ner"], []

        for word in tokens["text"]:
            char_ids = []
            for char in word:
                char_ids.append(self.char_alphabet.get_index(char))
            if len(char_ids) > self.config_data.max_char_length:
                char_ids = char_ids[: self.config_data.max_char_length]
            char_id_seqs.append(char_ids)

            word = self.normalize_func(word)
            word_ids.append(self.word_alphabet.get_index(word))

        for ner in ner_tags:
            ner_ids.append(self.ner_alphabet.get_index(ner))

        max_len = max([len(char_seq) for char_seq in char_id_seqs])
        self.max_char_length = max(self.max_char_length, max_len)

        self.train_instances_cache.append((word_ids, char_id_seqs, ner_ids))

    def epoch_finish_action(self, epoch):
        """
        At the end of each dataset_iteration, we perform the training,
        and set validation flags.

        :return:
        """
        counter = len(self.train_instances_cache)
        logger.info(f"Total number of ner_data: {counter}")

        lengths = sum(
            [len(instance[0]) for instance in self.train_instances_cache]
        )

        logger.info(f"Average sentence length: {(lengths / counter):0.3f}")

        train_err = 0.0
        train_total = 0.0

        start_time = time.time()
        self.model.train()

        # Each time we will clear and reload the train_instances_cache
        instances = self.train_instances_cache
        random.shuffle(self.train_instances_cache)
        data_iterator = torchtext.data.iterator.pool(
            instances,
            self.config_data.batch_size_tokens,
            key=lambda x: x.length(),  # length of word_ids
            batch_size_fn=_batch_size_fn,
            random_shuffler=torchtext.data.iterator.RandomShuffler(),
        )

        step = 0

        for batch in data_iterator:
            step += 1
            batch_data = self.get_batch_tensor(batch, device=self.device)
            word, char, labels, masks, lengths = batch_data

            self.optim.zero_grad()
            loss = self.model(word, char, labels, mask=masks)
            loss.backward()
            self.optim.step()

            num_inst = word.size(0)
            train_err += loss.item() * num_inst
            train_total += num_inst

            # update log
            if step % 200 == 0:
                logger.info(
                    f"Train: {step}, " f"loss: {(train_err / train_total):0.3f}"
                )

        logger.info(
            f"Epoch: {epoch}, steps: {step}, "
            f"loss: {(train_err / train_total):0.3f}, "
            f"time: {(time.time() - start_time):0.3f}s"
        )

        self.trained_epochs = epoch

        if epoch % self.config_model.decay_interval == 0:
            lr = self.config_model.learning_rate / (
                1.0 + self.trained_epochs * self.config_model.decay_rate
            )
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            logger.info(f"Update learning rate to {lr:0.3f}")

        self.request_eval()
        self.train_instances_cache.clear()

        if epoch >= self.config_data.num_epochs:
            self.request_stop_train()

    @torch.no_grad()
    def get_loss(self, instances: Iterator) -> float:
        """
        Compute the loss based on the validation data.

        Args:
            instances:

        Returns:

        """
        losses = 0
        val_data = list(instances)
        for i in tqdm(
            range(0, len(val_data), self.config_data.test_batch_size)
        ):
            b_data = val_data[i : i + self.config_data.test_batch_size]
            batch = self.get_batch_tensor(b_data, device=self.device)

            word, char, labels, masks, unused_lengths = batch
            loss = self.model(word, char, labels, mask=masks)
            losses += loss.item()

        mean_loss = losses / len(val_data)
        return mean_loss

    def post_validation_action(self, eval_result):
        """
        Log the evaluation results.

        Args:
            eval_result: The evaluation results.

        Returns:

        """

        if self.__past_dev_result is None or (
            eval_result["eval"]["f1"] > self.__past_dev_result["eval"]["f1"]
        ):
            self.__past_dev_result = eval_result
            logger.info("Validation f1 increased, saving model")
            self.__save_model_checkpoint()

        best_epoch = self.__past_dev_result["epoch"]
        acc, prec, rec, f1 = (
            self.__past_dev_result["eval"]["accuracy"],
            self.__past_dev_result["eval"]["precision"],
            self.__past_dev_result["eval"]["recall"],
            self.__past_dev_result["eval"]["f1"],
        )
        logger.info(
            f"Best val acc: {acc: 0.3f}, precision: {prec:0.3f}, "
            f"recall: {rec:0.3f}, F1: {f1:0.3f}, epoch={best_epoch}"
        )

        if "test" in self.__past_dev_result:
            acc, prec, rec, f1 = (
                self.__past_dev_result["test"]["accuracy"],
                self.__past_dev_result["test"]["precision"],
                self.__past_dev_result["test"]["recall"],
                self.__past_dev_result["test"]["f1"],
            )
            logger.info(
                f"Best test acc: {acc: 0.3f}, precision: {prec: 0.3f}, "
                f"recall: {rec: 0.3f}, F1: {f1: 0.3f}, "
                f"epoch={best_epoch}"
            )

    def finish(self, resources: Resources):  # pylint: disable=unused-argument
        """
        Releasing resources and saving models.

        Args:
            resources: The resources used by the training process.

        Returns:

        """
        if self.resource:
            keys_to_serializers = {}
            for key in resources.keys():
                # pylint: disable=consider-using-with
                if key == "model":
                    keys_to_serializers[key] = lambda x, y: pickle.dump(
                        x.state_dict(), open(y, "wb")
                    )
                else:
                    keys_to_serializers[key] = lambda x, y: pickle.dump(
                        x, open(y, "wb")
                    )

            self.resource.save(
                keys_to_serializers, output_dir=self.config_model.resource_dir
            )

        self.__save_model_checkpoint()

    def __save_model_checkpoint(self):
        states = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

        path = Path(self.config_model.model_path)
        if not Path(self.config_model.model_path).exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="wb") as f:
            torch.save(states, f)

    def load_model_checkpoint(self):
        """
        Load the model with a check pointer.

        Returns:

        """
        ckpt = torch.load(self.config_model.model_path)
        logger.info("restoring model from %s", self.config_model.model_path)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])

    def get_batch_tensor(
        self,
        data: List[Tuple[List[int], List[List[int]], List[int]]],
        device: Optional[torch.device] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Get the tensors to be fed into the model.

        Args:
            data: A list of tuple (word_ids, char_id_sequences, ner_ids)
            device: The device for the tensors.

        Returns:
            A tuple where

            - ``words``: A tensor of shape `[batch_size, batch_length]`
              representing the word ids in the batch
            - ``chars``: A tensor of shape
              `[batch_size, batch_length, char_length]` representing the char
              ids for each word in the batch
            - ``ners``: A tensor of shape `[batch_size, batch_length]`
              representing the ner ids for each word in the batch
            - ``masks``: A tensor of shape `[batch_size, batch_length]`
              representing the indices to be masked in the batch. 1 indicates
              no masking.
            - ``lengths``: A tensor of shape `[batch_size]` representing the
              length of each sentences in the batch
        """
        batch_size = len(data)
        batch_length = max([len(d[0]) for d in data])
        char_length = max(
            [max([len(charseq) for charseq in d[1]]) for d in data]
        )

        char_length = min(
            self.config_data.max_char_length,
            char_length + self.config_data.num_char_pad,
        )

        wid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)
        cid_inputs = np.empty(
            [batch_size, batch_length, char_length], dtype=np.int64
        )
        nid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)

        masks = np.zeros([batch_size, batch_length], dtype=np.float32)

        lengths = np.empty(batch_size, dtype=np.int64)

        for i, inst in enumerate(data):
            wids, cid_seqs, nids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = self.word_alphabet.pad_id
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
                cid_inputs[i, c, len(cids) :] = self.char_alphabet.pad_id
            cid_inputs[i, inst_size:, :] = self.char_alphabet.pad_id
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = self.ner_alphabet.pad_id
            # masks
            masks[i, :inst_size] = 1.0

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        ners = torch.from_numpy(nid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        return words, chars, ners, masks, lengths


def _batch_size_fn(new: Tuple, count: int, _: int):
    if count == 1:
        _batch_size_fn.max_length = 0  # type: ignore

    _batch_size_fn.max_length = max(  # type: ignore
        _batch_size_fn.max_length, len(new[0])  # type: ignore
    )
    elements = count * _batch_size_fn.max_length  # type: ignore
    return elements
