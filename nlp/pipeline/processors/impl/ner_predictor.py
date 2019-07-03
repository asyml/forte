from typing import Dict, List
from abc import abstractmethod
import os
import sys
import torch
import numpy as np
from nlp.pipeline.common.evaluation import Evaluator
from nlp.pipeline.processors.predictor import Predictor
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.models.NER.vocabulary_processor import Alphabet
from nlp.pipeline.data.readers.conll03_reader import CoNLL03Ontology


class CoNLLNERPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.word_alphabet, self.char_alphabet, self.chunk_alphabet, self.pos_alphabet, self.ner_alphabet = (
            None,
            None,
            None,
            None,
            None,
        )
        self.config_model = None
        self.config_data = None
        self.normalize_func = None
        self.embedding_dict = None
        self.embedding_dim = None
        self.device = None
        self.optim, self.trained_epochs = None, None
        self.add_cnt = 0
        self.input_cnt = 0

        self.train_instances_cache = []

        # TODO(haoransh): reconsider these hard-coded parameters
        self.context_type = "sentence"
        self.annotation_types = {
            "Token": ["chunk_tag", "pos_tag", "ner_tag"],
            "Sentence": [],  # span by default
        }
        self.batch_size = 3
        self.ner_ontology = CoNLL03Ontology
        self.component_name = "ner_predictor"

    def initialize(self, resource: Resources):
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
        self.device = resource.resources["device"]
        self.normalize_func = lambda x: self.config_data.digit_re.sub("0", x)

        self.trained_epochs = 0

    @torch.no_grad()
    def predict(self, data_batch: Dict):

        tokens = data_batch["Token"]
        offsets = data_batch["offset"]

        pred_tokens, instances = [], []
        for words, poses, chunks, ners in zip(
            tokens["text"],
            tokens["pos_tag"],
            tokens["chunk_tag"],
            tokens["ner_tag"],
        ):
            char_id_seqs = []
            word_ids = []
            pos_ids, chunk_ids, ner_ids = [], [], []
            for word in words:
                self.input_cnt += 1
                char_ids = []
                for char in word:
                    char_ids.append(self.char_alphabet.get_index(char))
                if len(char_ids) > self.config_data.max_char_length:
                    char_ids = char_ids[: self.config_data.max_char_length]
                char_id_seqs.append(char_ids)

                word = self.normalize_func(word)
                word_ids.append(self.word_alphabet.get_index(word))

            for pos in poses:
                pos_ids.append(self.pos_alphabet.get_index(pos))
            for chunk in chunks:
                chunk_ids.append(self.chunk_alphabet.get_index(chunk))
            for ner in ners:
                ner_ids.append(self.ner_alphabet.get_index(ner))
            instances.append(
                (word_ids, char_id_seqs, pos_ids, chunk_ids, ner_ids)
            )

        self.model.eval()
        batch_data = self.get_batch_tensor(instances, device=self.device)
        word, char, _, _, labels, masks, lengths = batch_data
        preds = self.model.decode(word, char, mask=masks)

        for i in range(len(tokens["text"])):
            sentence = tokens["text"][i]
            spans = tokens["span"][i]
            offset = offsets[i]
            poses, chunks, ners = (
                tokens["pos_tag"][i],
                tokens["chunk_tag"][i],
                tokens["ner_tag"][i],
            )
            for j in range(len(sentence)):
                predicted_tag = self.ner_alphabet.get_instance(preds[i][j])
                kwargs_i = {
                    "pos_tag": poses[j],
                    "chunk_tag": chunks[j],
                    "ner_tag": predicted_tag,
                }
                token = self.ner_ontology.Token(
                    self.component_name,
                    spans[j][0] + offset,
                    spans[j][1] + offset,
                )
                token.set_fields(**kwargs_i)
                pred_tokens.append(token)

        return pred_tokens

    def pack(self, data_pack: DataPack, *inputs):
        tokens = inputs[0]
        for i, token in enumerate(tokens):
            self.add_cnt += 1
            data_pack.add_entry(token)

        # TODO: input_cnt != add_cnt, but why?
        assert self.input_cnt == self.add_cnt

    def _record_fields(self, data_pack: DataPack):
        # data_pack.record_fields(
        #     [],
        #     self.component_name,
        #     self.ner_ontology.Sentence.__name__,
        # )
        data_pack.record_fields(
            ["chunk_tag", "pos_tag", "ner_tag"],
            self.component_name,
            self.ner_ontology.Token.__name__,
        )

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
                cid_inputs[i, c, len(cids) :] = self.char_alphabet.pad_id
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


class CoNLLNEREvaluator(Evaluator):
    def __init__(self, config):
        super().__init__(config)
        self.ner_ontology = CoNLL03Ontology
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores = {}

    def consume_next(self, pack: DataPack):
        tokens = pack.get_entries(self.ner_ontology.Token)
        ori_cnt, pred_cnt = 0, 0
        for token in tokens:
            if (
                token.component
                != "nlp.pipeline.data.readers.conll03_reader.CoNLL03Reader"
            ):
                pred_cnt += 1
            else:
                ori_cnt += 1
        opened_file = open(self.output_file, "w+")
        for pred_sentence, tgt_sentence in zip(
            pack.get_data(
                context_type="sentence",
                annotation_types={
                    "Token": {
                        "component": "ner_predictor",
                        "fields": ["chunk_tag", "pos_tag", "ner_tag"],
                    },
                    "Sentence": [],  # span by default
                },
            ),
            pack.get_data(
                context_type="sentence",
                annotation_types={
                    "Token": {"fields": ["chunk_tag", "pos_tag", "ner_tag"]},
                    "Sentence": [],  # span by default
                },
            ),
        ):

            pred_tokens, tgt_tokens = (
                pred_sentence["Token"],
                tgt_sentence["Token"],
            )
            for i in range(len(pred_tokens["text"])):
                w = pred_tokens["text"][i]
                p = pred_tokens["pos_tag"][i]
                ch = pred_tokens["chunk_tag"][i]
                tgt = tgt_tokens["ner_tag"][i]
                pred = pred_tokens["ner_tag"][i]

                opened_file.write(
                    "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred)
                )

            opened_file.write("\n")
        opened_file.close()
        os.system(
            "./conll03eval.v2 < %s > %s"
            % (self.output_file, self.score_file)
        )
        with open(self.score_file, "r") as fin:
            fin.readline()
            line = fin.readline()
            fields = line.split(";")
            acc = float(fields[0].split(":")[1].strip()[:-1])
            precision = float(fields[1].split(":")[1].strip()[:-1])
            recall = float(fields[2].split(":")[1].strip()[:-1])
            f1 = float(fields[3].split(":")[1].strip())
        self.scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def get_result(self):
        return self.scores
