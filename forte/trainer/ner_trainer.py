import logging
import random
import time
from typing import List, Tuple, Iterator, Optional

import numpy as np
import torch
from torch.optim import SGD
import torchtext
from tqdm import tqdm
from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.ontology import conll03_ontology as conll
from forte.trainer.base.base_trainer import BaseTrainer
from forte.models.ner import utils
from forte.models.ner.model_factory import BiRecurrentConvCRF

logger = logging.getLogger(__name__)


class CoNLLNERTrainer(BaseTrainer):
    def __init__(self):
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

        self.ontology = conll
        self.resource: Optional[Resources] = None

        self.train_instances_cache = []

        # Just for recording
        self.max_char_length = 0

        self.__past_dev_result = None

    def initialize(self, resource: Resources, configs: HParams):

        self.resource = resource
        # This reference is for saving the checkpoints

        self.word_alphabet = resource.get("word_alphabet")
        self.char_alphabet = resource.get("char_alphabet")
        self.ner_alphabet = resource.get("ner_alphabet")

        word_embedding_table = resource.get('word_embedding_table')

        self.config_model = configs.config_model
        self.config_data = configs.config_data

        self.normalize_func = utils.normalize_digit_word

        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        utils.set_random_seed(self.config_model.random_seed)

        self.model = BiRecurrentConvCRF(
            word_embedding_table,
            self.char_alphabet.size(),
            self.ner_alphabet.size(),
            self.config_model).to(device=self.device)

        self.optim = SGD(
            self.model.parameters(),
            lr=self.config_model.learning_rate,
            momentum=self.config_model.momentum,
            nesterov=True)

        self.trained_epochs = 0

        self.resource.update(model=self.model)

    def data_request(self):
        request_string = {
            "context_type": conll.Sentence,
            "request": {
                self.ontology.Token: ["ner_tag"],
                self.ontology.Sentence: [],  # span by default
            }
        }
        return request_string

    def consume(self, instance):
        tokens = instance["Token"]
        word_ids = []
        char_id_seqs = []
        ner_tags, ner_ids = tokens["ner_tag"], []

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

    def pack_finish_action(self, pack_count):
        pass

    def epoch_finish_action(self, epoch):
        """
        at the end of each dataset_iteration, we perform the training,
        and set validation flags
        :return:
        """
        counter = len(self.train_instances_cache)
        logger.info(f"Total number of ner_data: {counter}")

        lengths = \
            sum([len(instance[0]) for instance in self.train_instances_cache])

        logger.info(f"Average sentence length: {lengths / counter}")

        train_err = 0.0
        train_total = 0.0

        start_time = time.time()
        self.model.train()

        # Each time we will clear and reload the train_instances_cache
        instances = self.train_instances_cache
        random.shuffle(self.train_instances_cache)
        data_iterator = torchtext.data.iterator.pool(
            instances, self.config_data.batch_size_tokens,
            key=lambda x: x.length(),  # length of word_ids
            batch_size_fn=batch_size_fn,
            random_shuffler=torchtext.data.iterator.RandomShuffler())

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
                logger.info(f"Train: {step}, loss: {train_err / train_total}")

        logger.info(f"Epoch: {epoch}, steps: {step}, "
                    f"loss: {train_err / train_total}, "
                    f"time: {time.time() - start_time}s")

        self.trained_epochs = epoch

        if epoch % self.config_model.decay_interval == 0:
            lr = self.config_model.learning_rate / \
                 (1.0 + self.trained_epochs * self.config_model.decay_rate)
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            logger.info(f"Update learning rate to {lr}")

        self.request_eval()
        self.train_instances_cache.clear()

        if epoch >= self.config_data.num_epochs:
            self.request_stop_train()

    @torch.no_grad()
    def get_loss(self, instances: Iterator) -> float:
        losses = 0
        val_data = list(instances)
        for i in tqdm(
                range(0, len(val_data), self.config_data.test_batch_size)):
            b_data = val_data[i: i + self.config_data.test_batch_size]
            batch = self.get_batch_tensor(b_data, device=self.device)

            word, char, labels, masks, unused_lengths = batch
            loss = self.model(word, char, labels, mask=masks)
            losses += loss.item()

        mean_loss = losses / len(val_data)
        return mean_loss

    def post_validation_action(self, eval_result):
        if self.__past_dev_result is None or \
                (eval_result["eval"]["f1"] >
                 self.__past_dev_result["eval"]["f1"]):
            self.__past_dev_result = eval_result
            logger.info("Validation f1 increased, saving model")
            # self.save_model_checkpoint()

        best_epoch = self.__past_dev_result["epoch"]
        acc, prec, rec, f1 = (self.__past_dev_result["eval"]["accuracy"],
                              self.__past_dev_result["eval"]["precision"],
                              self.__past_dev_result["eval"]["recall"],
                              self.__past_dev_result["eval"]["f1"])
        logger.info(f"Best val acc: {acc: 0.3f}, precision: {prec:0.3f}, "
                    f"recall: {rec:0.3f}, F1: {f1:0.3f}, epoch={best_epoch}")

        if "test" in self.__past_dev_result:
            acc, prec, rec, f1 = (self.__past_dev_result["test"]["accuracy"],
                                  self.__past_dev_result["test"]["precision"],
                                  self.__past_dev_result["test"]["recall"],
                                  self.__past_dev_result["test"]["f1"])
            logger.info(f"Best test acc: {acc: 0.3f}, precision: {prec: 0.3f}, "
                        f"recall: {rec: 0.3f}, F1: {f1: 0.3f}, "
                        f"epoch={best_epoch}")

    def finish(self, resources: Resources):  # pylint: disable=unused-argument
        self.resource.save(output_dir=self.config_model.resource_dir)
        self.save_model_checkpoint()

    def save_model_checkpoint(self):
        states = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(states, self.config_model.model_path)

    def load_model_checkpoint(self):
        ckpt = torch.load(self.config_model.model_path)
        logger.info("restoring model from %s",
                    self.config_model.model_path)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])

    def get_batch_tensor(self, data: List, device=None):
        """

        Args:
            data: A list of tuple
              (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids)
            device: The device the tensor should be reside on.

        Returns:

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
                cid_inputs[i, c, len(cids):] = self.char_alphabet.pad_id
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


def batch_size_fn(new: Tuple, count: int, _: int):
    if count == 1:
        batch_size_fn.max_length = 0  # type: ignore

    batch_size_fn.max_length = max(  # type: ignore
        batch_size_fn.max_length, len(new[0]))  # type: ignore
    elements = count * batch_size_fn.max_length  # type: ignore
    return elements
