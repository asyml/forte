import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from texar.torch.hyperparams import HParams

from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.data import DataPack
from forte.data.datasets.conll import conll_utils
from forte.data.ontology import conll03_ontology as conll
from forte.processors.base import ProcessInfo
from forte.processors.base.batch_processor import FixedSizeBatchProcessor

logger = logging.getLogger(__name__)


class CoNLLNERPredictor(FixedSizeBatchProcessor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.word_alphabet, self.char_alphabet, self.ner_alphabet = (
            None, None, None)
        self.config_model = None
        self.config_data = None
        self.normalize_func = None
        self.device = None

        self.train_instances_cache = []

        self._ontology = conll
        self.define_context()

        self.batch_size = 3
        self.batcher = self.define_batcher()

    def define_context(self):
        self.context_type = self._ontology.Sentence

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Token: [],
            self._ontology.Sentence: [],
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self._ontology.EntityMention: ["ner_type", "span"],
        }
        return output_info

    def initialize(self, resource: Resources, configs: HParams):
        self.define_batcher()

        resource.load(configs.storage_path)

        self.word_alphabet = resource.resources["word_alphabet"]
        self.char_alphabet = resource.resources["char_alphabet"]
        self.ner_alphabet = resource.resources["ner_alphabet"]
        self.config_model = resource.resources["config_model"]
        self.config_data = resource.resources["config_data"]
        self.model = resource.resources["model"]
        self.device = resource.resources["device"]
        self.normalize_func = resource.resources['normalize_func']
        self.model.eval()

    @torch.no_grad()
    def predict(self, data_batch: Dict):

        tokens = data_batch["Token"]

        instances = []
        for words in tokens["text"]:
            char_id_seqs = []
            word_ids = []
            for word in words:
                char_ids = []
                for char in word:
                    char_ids.append(self.char_alphabet.get_index(char))
                if len(char_ids) > self.config_data.max_char_length:
                    char_ids = char_ids[: self.config_data.max_char_length]
                char_id_seqs.append(char_ids)

                word = self.normalize_func(word)
                word_ids.append(self.word_alphabet.get_index(word))

            instances.append(
                (word_ids, char_id_seqs)
            )

        self.model.eval()
        batch_data = self.get_batch_tensor(instances, device=self.device)
        word, char, masks, unused_lengths = batch_data
        preds = self.model.decode(word, char, mask=masks)

        pred: Dict = {"Token": {"ner_tag": [], "tid": []}}

        for i in range(len(tokens["tid"])):
            tids = tokens["tid"][i]
            ner_tags = []
            for j in range(len(tids)):
                ner_tags.append(self.ner_alphabet.get_instance(preds[i][j]))

            pred["Token"]["ner_tag"].append(np.array(ner_tags))
            pred["Token"]["tid"].append(np.array(tids))

        return pred

    def load_model_checkpoint(self, model_path=None):
        p = model_path if model_path is not None \
            else self.config_model.model_path
        ckpt = torch.load(p)
        logger.info(
            "restoring NER model from %s", self.config_model.model_path)
        self.model.load_state_dict(ckpt["model"])

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        """
        Write the prediction results back to datapack. by writing the predicted
        ner_tag to the original tokens.
        """
        if output_dict is None:
            return

        # Overwrite the tokens in the data_pack

        current_entity_mention: Tuple[int, str] = (-1, "None")

        for i in range(len(output_dict["Token"]["tid"])):
            # an instance
            for j in range(len(output_dict["Token"]["tid"][i])):
                tid = output_dict["Token"]["tid"][i][j]

                orig_token: conll.Token = data_pack.get_entry(  # type: ignore
                    tid)
                ner_tag: str = output_dict["Token"]["ner_tag"][i][j]

                orig_token.set_fields(ner_tag=ner_tag)

                token = orig_token
                token_ner = token.get_field("ner_tag")
                if token_ner[0] == "B":
                    current_entity_mention = (
                        token.span.begin,
                        token_ner[2:],
                    )
                elif token_ner[0] == "I":
                    continue
                elif token_ner[0] == "O":
                    continue

                elif token_ner[0] == "E":
                    if token_ner[2:] != current_entity_mention[1]:
                        continue

                    kwargs_i = {"ner_type": current_entity_mention[1]}
                    entity = self._ontology.EntityMention(
                        data_pack,
                        current_entity_mention[0],
                        token.span.end,
                    )
                    entity.set_fields(**kwargs_i)
                    data_pack.add_or_get_entry(entity)
                elif token_ner[0] == "S":
                    current_entity_mention = (
                        token.span.begin,
                        token_ner[2:],
                    )
                    kwargs_i = {"ner_type": current_entity_mention[1]}
                    entity = self._ontology.EntityMention(
                        data_pack,
                        current_entity_mention[0],
                        token.span.end,
                    )
                    entity.set_fields(**kwargs_i)
                    data_pack.add_or_get_entry(entity)

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

        masks = np.zeros([batch_size, batch_length], dtype=np.float32)

        lengths = np.empty(batch_size, dtype=np.int64)

        for i, inst in enumerate(data):
            wids, cid_seqs = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = self.word_alphabet.pad_id
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
                cid_inputs[i, c, len(cids):] = self.char_alphabet.pad_id
            cid_inputs[i, inst_size:, :] = self.char_alphabet.pad_id
            masks[i, :inst_size] = 1.0

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        return words, chars, masks, lengths

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        hparams_dict = {
            'storage_path': None,
        }
        return hparams_dict


class CoNLLNEREvaluator(Evaluator):
    def __init__(self, config: Optional[HParams] = None):
        super().__init__(config)
        self._ontology = conll
        self.test_component = CoNLLNERPredictor().component_name
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores: Dict[str, float] = {}

    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack):
        pred_getdata_args = {
            "context_type": "sentence",
            "requests": {
                conll.Token: {
                    "fields": ["ner_tag"],
                },
                conll.Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": "sentence",
            "requests": {
                conll.Token: {
                    "fields": ["chunk_tag", "pos_tag", "ner_tag"]},
                conll.Sentence: [],  # span by default
            }
        }

        conll_utils.write_tokens_to_file(pred_pack=pred_pack,
                                         pred_request=pred_getdata_args,
                                         refer_pack=refer_pack,
                                         refer_request=refer_getdata_args,
                                         output_filename=self.output_file)
        os.system(
            "./conll03eval.v2 < %s > %s" % (self.output_file, self.score_file)
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
