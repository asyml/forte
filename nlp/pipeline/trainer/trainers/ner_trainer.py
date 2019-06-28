from abc import abstractmethod
from typing import List, Tuple
import logging
import random
import re
import sys
import numpy as np
import time
import torch
import torchtext
from torch.optim import SGD
from typing import Dict, Any, Iterator
from nlp.pipeline.io.data_pack import DataPack
from nlp.pipeline.models.NER.vocabulary_processor import Alphabet
from nlp.pipeline.models.NER.model_factory import BiRecurrentConvCRF
from nlp.pipeline.trainer import Trainer


class CoNLLNERTrainer(Trainer):
    def __init__(self, resources, model, optim, **kwargs):
        super().__init__()

        # TODO(haoransh): refine the interface here
        self.kwargs = kwargs
        self.word_alphabet: Alphabet = kwargs['word_alphabet']
        self.char_alphabet: Alphabet = kwargs['char_alphabet']
        self.chunk_alphabet: Alphabet = kwargs['chunk_alphabet']
        self.pos_alphabet: Alphabet = kwargs['pos_alphabet']
        self.ner_alphabet: Alphabet = kwargs['ner_alphabet']
        self.config_model = kwargs['config_model']
        self.config_data = kwargs['config_data']
        self.embedding_dict = kwargs['embedding_dict']
        self.embedding_dim = kwargs['embedding_dim']

        # TODO(haoransh): polish these hyper-parameters
        self.max_char_length = 45
        self.num_char_pad = 2

        self.normalize_digit = True
        digit_re = re.compile(r"\d")

        self.normalize_func = lambda x: digit_re.sub("0", x)
        model = BiRecurrentConvCRF(self.word_alphabet,
                                   self.char_alphabet,
                                   self.ner_alphabet,
                                   self.embedding_dict,
                                   self.embedding_dim)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = model.to_device(self.device)
        self.optim = SGD(
            self.model.parameters(),
            lr=self.config_model.learning_rate,
            momentum=self.config_model.momentum,
            nesterov=True,
        )

        self.trained_epochs = 0

    def update(self):

        pass

    # write data into data pack
    # eval(pack, goldpack)

    def process(self, input_pack: Iterator[DataPack]):
        counter = 0
        word_alphabet = self.word_alphabet
        char_alphabet = self.char_alphabet
        chunk_alphabet = self.chunk_alphabet
        pos_alphabet = self.pos_alphabet
        ner_alphabet = self.ner_alphabet

        instances = []
        lengths = 0
        max_char_length = 0

        for data_pack in input_pack:
            for instance in data_pack.get_data(
                    context_type="sentence",
                    annotation_types={
                        "Token": ["chunk_tag", "pos_tag", "ner_tag"],
                        "Sentence": [],  # span by default
                    },
            ):
                counter += 1
                tokens = instance['Token']
                word_ids = []
                char_seqs, char_id_seqs = [], []
                ner_tags, ner_ids = tokens["ner_tag"], []
                pos_tags, pos_ids = tokens["pos_tag"], []
                chunk_tags, chunk_ids = tokens["chunk_tag"], []

                for word in tokens["text"]:
                    char_ids = []
                    for char in word:
                        char_ids.append(char_alphabet.get_index(char))
                    if len(char_ids) > self.max_char_length:
                        char_ids = char_ids[: self.max_char_length]
                    char_id_seqs.append(char_ids)

                    word = self.normalize_func(word)
                    word_ids.append(word_alphabet.get_index(word))

                for pos in pos_tags:
                    pos_ids.append(pos_alphabet.get_index(pos))
                for chunk in chunk_tags:
                    chunk_ids.append(chunk_alphabet.get_index(chunk))
                for ner in ner_tags:
                    ner_ids.append(ner_alphabet.get_index(ner))

                inst_size = len(word_ids)

                instances.append(
                    (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids))
                max_len = max([len(char_seq) for char_seq in char_seqs])
                max_char_length = max(max_char_length, max_len)

                lengths += inst_size

        logging.info("Total number of data: %d" % counter)
        logging.info("average sentence length: %f" % (lengths / counter))

        train_err = 0.0
        train_total = 0.0

        start_time = time.time()
        num_back = 0
        self.model.train()

        random.shuffle(instances)
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
            if bid % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = "train: %d loss: %.4f" % (
                    bid,
                    train_err / train_total,
                )
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print(
            "train: %d loss: %.4f, time: %.2fs"
            % (bid, train_err / train_total, time.time() - start_time)
        )

        self.trained_epochs += 1

        if self.trained_epochs % self.config_model.decay_interval == 0:
            lr = self.config_model.learning_rate / (1.0 + self.trained_epochs
                                                    * self.config_model.decay_rate)
            for param_group in self.optim.param_groups:
                param_group["lr"] = lr

    def get_batch_tensor(self, data: List, device=None):
        """

        :param data: A list of quintuple
            (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids
        :param device:
        :return:
        """
        batch_size = len(data)
        batch_length = max([len(d[0]) for d in data])
        char_length = max([max([len(charseq) for charseq in d[1]]) for d in data])

        char_length = min(self.max_char_length, char_length + self.num_char_pad)

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
            wid_inputs[i, inst_size:] = self.word_alphabet.bos_id
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
                cid_inputs[i, c, len(cids) :] = \
                    self.char_alphabet.pad_id
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


def batch_size_fn(new: Tuple, count: int, size_so_far: int):
    if count == 1:
        batch_size_fn.max_length = 0
    batch_size_fn.max_length = max(batch_size_fn.max_length, len(new[0]))
    elements = count * batch_size_fn.max_length
    return elements


