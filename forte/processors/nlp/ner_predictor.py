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
from typing import Dict, List, Optional, Tuple
import numpy as np

from forte.common import ProcessorConfigError, ResourceError
from forte.utils import create_import_error_msg
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.models.ner import utils
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.processors.base.batch_processor import RequestPackingProcessor
from ft.onto.base_ontology import Token, EntityMention

try:
    import torch
except ImportError as e:
    raise ImportError(
        create_import_error_msg("torch", "models", "ner predictor")
    ) from e


logger = logging.getLogger(__name__)

__all__ = [
    "CoNLLNERPredictor",
]


class CoNLLNERPredictor(RequestPackingProcessor):
    """
    An Named Entity Recognizer trained according to `Ma, Xuezhe, and Eduard
    Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf."
    <https://arxiv.org/abs/1603.01354>`_.

    Note that to use :class:`CoNLLNERPredictor`, the :attr:`ontology` of
    :class:`~forte.pipeline.Pipeline` must be an ontology that include
    :class:`ft.onto.base_ontology.Token` and :class:`ft.onto.base_ontology.Sentence`.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.word_alphabet, self.char_alphabet, self.ner_alphabet = (
            None,
            None,
            None,
        )
        self.resource = None
        self.config_model = None
        self.config_data = None
        self.normalize_func = None
        self.device = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        self.resource = resources
        self.config_model = configs.config_model
        self.config_data = configs.config_data

        resource_path = configs.config_model.resource_dir

        keys = {
            "word_alphabet",
            "char_alphabet",
            "ner_alphabet",
            "word_embedding_table",
        }

        missing_keys = list(keys.difference(self.resource.keys()))

        self.resource.load(keys=missing_keys, path=resource_path)

        self.word_alphabet = resources.get("word_alphabet")
        self.char_alphabet = resources.get("char_alphabet")
        self.ner_alphabet = resources.get("ner_alphabet")
        word_embedding_table = resources.get("word_embedding_table")

        if resources.get("device"):
            self.device = resources.get("device")
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.normalize_func = utils.normalize_digit_word

        if "model" not in self.resource.keys():

            def load_model(path):
                if (
                    self.word_alphabet is None
                    or self.char_alphabet is None
                    or self.ner_alphabet is None
                ):
                    raise ResourceError(
                        "Error when configuring the predictor, alphabets "
                        "loaded from the resources are not initialized."
                    )

                model = BiRecurrentConvCRF(
                    word_embedding_table,
                    self.char_alphabet.size(),
                    self.ner_alphabet.size(),
                    self.config_model,
                )

                if os.path.exists(path):
                    with open(path, "rb") as f:
                        weights = torch.load(f, map_location=self.device)
                        model.load_state_dict(weights)
                return model

            self.resource.load(keys={"model": load_model}, path=resource_path)

        self.model = resources.get("model")
        self.model.to(self.device)
        self.model.eval()

        utils.set_random_seed(self.config_model.random_seed)

    @torch.no_grad()
    def predict(
        self, data_batch: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        if self.config_data is None:
            raise ProcessorConfigError(
                "Data configuration for the predictor is not found."
            )

        if self.model is None:
            raise ProcessorConfigError("Model for the predictor is not set.")

        if self.normalize_func is None:
            raise ProcessorConfigError(
                "The normalizing function for the predictor is not set."
            )

        if (
            self.word_alphabet is None
            or self.ner_alphabet is None
            or self.word_alphabet is None
            or self.char_alphabet is None
        ):
            raise ProcessorConfigError(
                "Error when configuring the predictor, alphabets are not initialized."
            )

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
        if self.config_model is None:
            raise ProcessorConfigError(
                "Model configuration for the predictor is not found."
            )

        if self.model is None:
            raise ProcessorConfigError("Model is not set for the predictor.")

        p = (
            model_path
            if model_path is not None
            else self.config_model.model_path
        )
        ckpt = torch.load(p, map_location=self.device)
        logger.info(f"Restoring NER model from {self.config_model.model_path}")
        self.model.load_state_dict(ckpt["model"])

    def pack(
        self,
        pack: DataPack,
        predict_results: Dict[str, Dict[str, List[str]]],
        _: Optional[Annotation] = None,
    ):
        """
        Write the prediction results back to datapack. by writing the predicted
        ner to the original tokens.
        """

        if predict_results is None:
            return

        current_entity_mention: Tuple[int, str] = (-1, "None")

        for i in range(len(predict_results["Token"]["tid"])):
            # an instance
            for j in range(len(predict_results["Token"]["tid"][i])):
                tid: int = predict_results["Token"]["tid"][i][j]  # type: ignore

                orig_token: Token = pack.get_entry(tid)  # type: ignore
                ner_tag: str = predict_results["Token"]["ner"][i][j]

                orig_token.ner = ner_tag

                token = orig_token
                token_ner = token.ner
                assert isinstance(token_ner, str)
                if token_ner[0] == "B":
                    current_entity_mention = (token.begin, token_ner[2:])
                elif token_ner[0] == "I":
                    continue
                elif token_ner[0] == "O":
                    continue

                elif token_ner[0] == "E":
                    if token_ner[2:] != current_entity_mention[1]:
                        continue

                    entity = EntityMention(
                        pack, current_entity_mention[0], token.end
                    )
                    entity.ner_type = current_entity_mention[1]
                elif token_ner[0] == "S":
                    current_entity_mention = (token.begin, token_ner[2:])
                    entity = EntityMention(
                        pack, current_entity_mention[0], token.end
                    )
                    entity.ner_type = current_entity_mention[1]

    def get_batch_tensor(
        self,
        data: List[Tuple[List[int], List[List[int]]]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if self.config_data is None:
            raise ProcessorConfigError(
                "Data configuration for the predictor is not found."
            )

        if self.config_model is None:
            raise ProcessorConfigError(
                "Model configuration for the predictor is not found."
            )

        if (
            self.word_alphabet is None
            or self.ner_alphabet is None
            or self.word_alphabet is None
            or self.char_alphabet is None
        ):
            raise ProcessorConfigError(
                "Error when configuring the predictor, alphabets are not initialized."
            )

        batch_size = len(data)
        batch_length = max(len(d[0]) for d in data)
        char_length = max(max(len(charseq) for charseq in d[1]) for d in data)

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
                cid_inputs[i, c, len(cids) :] = self.char_alphabet.pad_id
            cid_inputs[i, inst_size:, :] = self.char_alphabet.pad_id
            masks[i, :inst_size] = 1.0

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        return words, chars, masks, lengths

    # TODO: change this to manageable size
    @classmethod
    def default_configs(cls):
        r"""Default config for NER Predictor"""
        return {
            "config_data": {
                "train_path": "",
                "val_path": "",
                "test_path": "",
                "num_epochs": 200,
                "batch_size_tokens": 512,
                "test_batch_size": 16,
                "max_char_length": 45,
                "num_char_pad": 2,
            },
            "config_model": {
                "output_hidden_size": 128,
                "dropout_rate": 0.3,
                "word_emb": {"dim": 100},
                "char_emb": {"dim": 30, "initializer": {"type": "normal_"}},
                "char_cnn_conv": {
                    "in_channels": 30,
                    "out_channels": 30,
                    "kernel_size": 3,
                    "padding": 2,
                },
                "bilstm_sentence_encoder": {
                    "rnn_cell_fw": {
                        "input_size": 130,
                        "type": "LSTMCell",
                        "kwargs": {"num_units": 128},
                    },
                    "rnn_cell_share_config": "yes",
                    "output_layer_fw": {"num_layers": 0},
                    "output_layer_share_config": "yes",
                },
                "learning_rate": 0.01,
                "momentum": 0.9,
                "decay_interval": 1,
                "decay_rate": 0.05,
                "random_seed": 1234,
                "initializer": {"type": "xavier_uniform_"},
                "model_path": "",
                "resource_dir": "",
            },
            "batcher": {
                "batch_size": 16,
                "context_type": "ft.onto.base_ontology.Sentence",
                "requests": {
                    "ft.onto.base_ontology.Token": [],
                    "ft.onto.base_ontology.Sentence": [],
                },
            },
        }
