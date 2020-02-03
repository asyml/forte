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

# pylint: disable=logging-fstring-interpolation
import logging
import os
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from texar.torch.hyperparams import HParams

from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.common.types import DataRequest
from forte.data.datasets.conll import conll_utils
from forte.data.ontology import Annotation
from forte.models.ner import utils
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention

logger = logging.getLogger(__name__)


class CoNLLNERPredictor(FixedSizeBatchProcessor):
    """
       An Named Entity Recognizer trained according to `Ma, Xuezhe, and Eduard
       Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf."
       <https://arxiv.org/abs/1603.01354>`_.

       Note that to use :class:`CoNLLNERPredictor`, the :attr:`ontology` of
       :class:`Pipeline` must be an ontology that include
       ``ft.onto.base_ontology.Token`` and ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.word_alphabet, self.char_alphabet, self.ner_alphabet = (
            None, None, None)
        self.resource = None
        self.config_model = None
        self.config_data = None
        self.normalize_func = None
        self.device = None

        self.train_instances_cache = []

        self.batch_size = 3
        self.batcher = self.define_batcher()

    def define_context(self) -> Type[Annotation]:
        return Sentence

    def _define_input_info(self) -> DataRequest:
        input_info: DataRequest = {
            Token: [],
            Sentence: [],
        }
        return input_info

    def initialize(self, resource: Resources, configs: HParams):

        self.define_batcher()

        self.resource = resource
        self.config_model = configs.config_model
        self.config_data = configs.config_data

        resource_path = configs.config_model.resource_dir

        keys = {"word_alphabet", "char_alphabet", "ner_alphabet",
                "word_embedding_table"}

        missing_keys = list(keys.difference(self.resource.keys()))

        self.resource.load(keys=missing_keys, path=resource_path)

        self.word_alphabet = resource.get("word_alphabet")
        self.char_alphabet = resource.get("char_alphabet")
        self.ner_alphabet = resource.get("ner_alphabet")
        word_embedding_table = resource.get("word_embedding_table")

        if resource.get("device"):
            self.device = resource.get("device")
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')

        self.normalize_func = utils.normalize_digit_word

        if "model" not in self.resource.keys():
            def load_model(path):
                model = BiRecurrentConvCRF(
                    word_embedding_table, self.char_alphabet.size(),
                    self.ner_alphabet.size(), self.config_model)

                if os.path.exists(path):
                    with open(path, "rb") as f:
                        weights = torch.load(f, map_location=self.device)
                        model.load_state_dict(weights)
                return model

            self.resource.load(keys={"model": load_model}, path=resource_path)

        self.model = resource.get("model")
        self.model.to(self.device)
        self.model.eval()

        utils.set_random_seed(self.config_model.random_seed)

    @torch.no_grad()
    def predict(self, data_batch: Dict[str, Dict[str, List[str]]]) \
            -> Dict[str, Dict[str, List[np.array]]]:
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

            instances.append((word_ids, char_id_seqs))

        self.model.eval()
        batch_data = self.get_batch_tensor(instances, device=self.device)
        word, char, masks, unused_lengths = batch_data
        preds = self.model.decode(word, char, mask=masks)

        pred: Dict = {"Token": {"ner": [], "tid": []}}

        for i in range(len(tokens["tid"])):
            tids = tokens["tid"][i]
            ner_tags = []
            for j in range(len(tids)):
                ner_tags.append(self.ner_alphabet.get_instance(preds[i][j]))

            pred["Token"]["ner"].append(np.array(ner_tags))
            pred["Token"]["tid"].append(np.array(tids))

        return pred

    def load_model_checkpoint(self, model_path=None):
        p = model_path if model_path is not None \
            else self.config_model.model_path
        ckpt = torch.load(p, map_location=self.device)
        logger.info(f"Restoring NER model from {self.config_model.model_path}")
        self.model.load_state_dict(ckpt["model"])

    def pack(self, data_pack: DataPack,
             output_dict: Optional[Dict[str, Dict[str, List[str]]]] = None):
        """
        Write the prediction results back to datapack. by writing the predicted
        ner to the original tokens.
        """

        if output_dict is None:
            return

        current_entity_mention: Tuple[int, str] = (-1, "None")

        for i in range(len(output_dict["Token"]["tid"])):
            # an instance
            for j in range(len(output_dict["Token"]["tid"][i])):
                tid: int = output_dict["Token"]["tid"][i][j]  # type: ignore

                orig_token: Token = data_pack.get_entry(tid)  # type: ignore # pylint: disable=line-too-long
                ner_tag: str = output_dict["Token"]["ner"][i][j]

                orig_token.set_fields(ner=ner_tag)

                token = orig_token
                token_ner = token.get_field("ner")
                if token_ner[0] == "B":
                    current_entity_mention = (token.span.begin, token_ner[2:])
                elif token_ner[0] == "I":
                    continue
                elif token_ner[0] == "O":
                    continue

                elif token_ner[0] == "E":
                    if token_ner[2:] != current_entity_mention[1]:
                        continue

                    kwargs_i = {"ner_type": current_entity_mention[1]}
                    entity = EntityMention(data_pack,
                                           current_entity_mention[0],
                                           token.span.end)
                    entity.set_fields(**kwargs_i)
                    data_pack.add_or_get_entry(entity)
                elif token_ner[0] == "S":
                    current_entity_mention = (token.span.begin, token_ner[2:])
                    kwargs_i = {"ner_type": current_entity_mention[1]}
                    entity = EntityMention(data_pack, current_entity_mention[0],
                                           token.span.end)
                    entity.set_fields(**kwargs_i)
                    data_pack.add_or_get_entry(entity)

    def get_batch_tensor(
            self, data: List[Tuple[List[int], List[List[int]]]],
            device: Optional[torch.device] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the tensors to be fed into the model.

        Args:
            data: A list of tuple (word_ids, char_id_sequences)
            device: The device for the tensors.

        Returns:
            A tuple where

            - ``words``: A tensor of shape `[batch_size, batch_length]`
              representing the word ids in the batch
            - ``chars``: A tensor of shape
              `[batch_size, batch_length, char_length]` representing the char
              ids for each word in the batch
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
    def default_configs():
        """
        This defines a basic config structure
        :return:
        """
        hparams_dict = {
            'storage_path': None,
        }
        return hparams_dict


class CoNLLNEREvaluator(Evaluator):
    def __init__(self, config: Optional[HParams] = None):
        super().__init__(config)
        self.test_component = CoNLLNERPredictor().component_name
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores: Dict[str, float] = {}

    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack):
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["ner"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["chunk", "pos", "ner"]},
                Sentence: [],  # span by default
            }
        }

        conll_utils.write_tokens_to_file(pred_pack=pred_pack,
                                         pred_request=pred_getdata_args,
                                         refer_pack=refer_pack,
                                         refer_request=refer_getdata_args,
                                         output_filename=self.output_file)
        os.system(f"./conll03eval.v2 < {self.output_file} > {self.score_file}")
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
