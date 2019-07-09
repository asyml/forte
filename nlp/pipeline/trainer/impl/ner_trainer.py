import logging
import random
import time
from typing import Iterator
from typing import List, Tuple

import numpy as np
import torch
import torchtext
from tqdm import tqdm

from nlp.pipeline.models.NER.vocabulary_processor import Alphabet
from nlp.pipeline.trainer.base_trainer import BaseTrainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CoNLLNERTrainer(BaseTrainer):
    def __init__(self, config=None):
        super().__init__(config)

        self.model = None
        self.word_alphabet = None
        self.char_alphabet = None
        self.chunk_alphabet = None
        self.pos_alphabet = None
        self.ner_alphabet = None
        self.config_model = None
        self.config_data = None
        self.normalize_func = None
        self.embedding_dict = None
        self.embedding_dim = None
        self.device = None
        self.optim, self.trained_epochs = None, None

        self.train_instances_cache = []
        self.max_char_length = 0

        self.__past_dev_result = None

    def initialize(self, resource):

        self.word_alphabet: Alphabet = resource.resources["word_alphabet"]
        self.char_alphabet: Alphabet = resource.resources["char_alphabet"]
        self.chunk_alphabet: Alphabet = resource.resources["chunk_alphabet"]
        self.pos_alphabet: Alphabet = resource.resources["pos_alphabet"]
        self.ner_alphabet: Alphabet = resource.resources["ner_alphabet"]
        self.config_model = resource.resources["config_model"]
        self.config_data = resource.resources["config_data"]
        self.embedding_dict = resource.resources["embedding_dict"]
        self.embedding_dim = resource.resources["embedding_dim"]
        self.model = resource.resources["model"]
        self.optim = resource.resources["optim"]
        self.device = resource.resources["device"]

        self.normalize_func = lambda x: self.config_data.digit_re.sub("0", x)

        self.trained_epochs = 0

    def data_request(self):
        request_string = {
            "context_type": "sentence",
            "annotation_types": {
                "Token": ["chunk_tag", "pos_tag", "ner_tag"],
                "Sentence": [],  # span by default
            },
        }
        return request_string

    def process(self, instance):
        tokens = instance["Token"]
        word_ids = []
        char_id_seqs = []
        ner_tags, ner_ids = tokens["ner_tag"], []
        pos_tags, pos_ids = tokens["pos_tag"], []
        chunk_tags, chunk_ids = tokens["chunk_tag"], []

        for word in tokens["text"]:
            char_ids = []
            for char in word:
                char_ids.append(self.char_alphabet.get_index(char))
            if len(char_ids) > self.config_data.max_char_length:
                char_ids = char_ids[: self.config_data.max_char_length]
            char_id_seqs.append(char_ids)

            word = self.normalize_func(word)
            word_ids.append(self.word_alphabet.get_index(word))

        for pos in pos_tags:
            pos_ids.append(self.pos_alphabet.get_index(pos))
        for chunk in chunk_tags:
            chunk_ids.append(self.chunk_alphabet.get_index(chunk))
        for ner in ner_tags:
            ner_ids.append(self.ner_alphabet.get_index(ner))

        max_len = max([len(char_seq) for char_seq in char_id_seqs])
        self.max_char_length = max(self.max_char_length, max_len)

        self.train_instances_cache.append(
            (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids)
        )

    def pack_finish_action(self, pack_count):

        pass

    def epoch_finish_action(self, epoch):
        """
        at the end of each dataset_iteration, we perform the training,
        and set validation flags
        :return:
        """
        counter = len(self.train_instances_cache)
        logger.info("Total number of ner_data: %d" % counter)
        lengths = sum(
            [len(instance[0]) for instance in self.train_instances_cache]
        )
        logger.info("average sentence length: %f" % (lengths / counter))

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
            batch_size_fn=batch_size_fn,
            random_shuffler=torchtext.data.iterator.RandomShuffler(),
        )
        bid = 0

        for batch in data_iterator:
            bid += 1
            batch_data = self.get_batch_tensor(batch, device=self.device)
            word, char, _, _, labels, masks, lengths = batch_data

            self.optim.zero_grad()
            loss = self.model(word, char, labels, mask=masks)
            loss.backward()
            self.optim.step()

            num_inst = word.size(0)
            train_err += loss.item() * num_inst
            train_total += num_inst

            # update log
            if bid % 200 == 0:
                log_info = "train: %d loss: %.4f" % (
                    bid,
                    train_err / train_total,
                )
                logger.info(log_info)

        logger.info(
            "Epoch: %d train: %d loss: %.4f, time: %.2fs"
            % (epoch, bid, train_err / train_total, time.time() - start_time)
        )

        self.trained_epochs = epoch

        if epoch % self.config_model.decay_interval == 0:
            lr = self.config_model.learning_rate / (
                    1.0 + self.trained_epochs * self.config_model.decay_rate
            )
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            logger.info(f"update learning rate to {lr}")

        self.request_eval()
        self.train_instances_cache.clear()

        if epoch >= self.config_data.num_epochs:
            self.request_stop_train()

    @torch.no_grad()
    def get_loss(self, instances: Iterator) -> float:
        losses = 0
        val_data = list(instances)
        for i in tqdm(
                range(0, len(val_data), self.config_data.test_batch_size)
        ):
            b_data = val_data[i: i + self.config_data.test_batch_size]
            batch = self.get_batch_tensor(b_data, device=self.device)

            word, char, pos, chunk, labels, masks, lengths = batch
            loss = self.model(word, char, labels, mask=masks)
            losses += loss.item()

        mean_loss = losses / len(val_data)
        return mean_loss

    def post_validation_action(self, eval_result):
        # TODO: generalize this method into base trainer
        if (
                not self.__past_dev_result
                or eval_result["eval"]["f1"] > self.__past_dev_result["eval"][
            "f1"]
        ):
            self.__past_dev_result = eval_result
            logger.info("validation f1 increased, saving model")
            self.save_model_checkpoint()

        best_epoch = self.__past_dev_result["epoch"]
        acc, prec, rec, f1 = (
            self.__past_dev_result["eval"]["accuracy"],
            self.__past_dev_result["eval"]["precision"],
            self.__past_dev_result["eval"]["recall"],
            self.__past_dev_result["eval"]["f1"],
        )
        logger.info(
            f"best val acc: {acc}, precision: {prec}, recall: {rec}, "
            f"F1: {f1} % (epoch: {best_epoch})"
        )

        acc, prec, rec, f1 = (
            self.__past_dev_result["test"]["accuracy"],
            self.__past_dev_result["test"]["precision"],
            self.__past_dev_result["test"]["recall"],
            self.__past_dev_result["test"]["f1"],
        )
        logger.info(
            f"best test acc: {acc}, precision: {prec}, recall: {rec}, "
            f"F1: {f1} % (epoch: {best_epoch})"
        )

    def finish(self):
        self.save_model_checkpoint()

    def save_model_checkpoint(self):
        states = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(states, self.config_model.model_path)

    def load_model_checkpoint(self):
        ckpt = torch.load(self.config_model.model_path)
        print("restoring model from {}".format(self.config_model.model_path))
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])

    def get_batch_tensor(self, data: List, device=None):
        """

        :param data: A list of quintuple
            (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids
        :param device:
        :return:
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
        pid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)
        chid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)
        nid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)

        masks = np.zeros([batch_size, batch_length], dtype=np.float32)

        lengths = np.empty(batch_size, dtype=np.int64)

        for i, inst in enumerate(data):
            wids, cid_seqs, pids, chids, nids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = self.word_alphabet.pad_id
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
                cid_inputs[i, c, len(cids):] = self.char_alphabet.pad_id
            cid_inputs[i, inst_size:, :] = self.char_alphabet.pad_id
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = self.pos_alphabet.pad_id
            # chunk ids
            chid_inputs[i, :inst_size] = chids
            chid_inputs[i, inst_size:] = self.chunk_alphabet.pad_id
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = self.ner_alphabet.pad_id
            # masks
            masks[i, :inst_size] = 1.0

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        chunks = torch.from_numpy(chid_inputs).to(device)
        ners = torch.from_numpy(nid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        return words, chars, pos, chunks, ners, masks, lengths


def batch_size_fn(new: Tuple, count: int, _: int):
    if count == 1:
        batch_size_fn.max_length = 0
    batch_size_fn.max_length = max(batch_size_fn.max_length, len(new[0]))
    elements = count * batch_size_fn.max_length
    return elements
