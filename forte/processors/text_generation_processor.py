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

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import texar.torch as tx
from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import MultiPack
from forte.data import MultiPackLink
from forte.data.batchers import (
    ProcessingBatcher, FixedSizeMultiPackProcessingBatcher)
from forte.common.types import DataRequest
from forte.processors.base.batch_processor import MultiPackBatchProcessor
from ft.onto.base_ontology import Sentence

logger = logging.getLogger(__name__)

__all__ = [
    "TextGenerationProcessor"
]


class TextGenerationProcessor(MultiPackBatchProcessor):

    def __init__(self):
        super().__init__()
        self.input_pack_name = None
        self.output_pack_name = None

        self.word_processor = None
        self.model = None

        self.batch_size = 6
        self._get_helper = None

        self.max_decoding_length = None
        self.temperature = None
        self.top_k = None
        self.top_p = None
        self.device = None
        self.define_context()

    def _define_input_info(self) -> DataRequest:
        # pylint: disable=no-self-use
        return {}

    def define_context(self):
        # pylint: disable=no-self-use
        self.context_type = Sentence

    def define_batcher(self) -> ProcessingBatcher:
        # pylint: disable=no-self-use
        return FixedSizeMultiPackProcessingBatcher()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        """
        Args:
            resource:
            configs: A config with the following keys:
                * input_pack_name: specify the input pack name of the MultiPack
                  to be processed
                * output_pack_name: specify the output pack name of the
                  MultiPack to be processed
                * max_decoding_length: the maximum decoding length.
                * top_k
                * top_p
                * temperature

        Returns:
        """
        super().initialize(resource, configs)

        if configs is not None:
            self.input_pack_name = configs.input_pack_name
            self.output_pack_name = configs.output_pack_name

            self.max_decoding_length = configs.max_decoding_length
            self.temperature = configs.temperature
            self.top_k = configs.top_k
            self.top_p = configs.top_p
            self.model = tx.modules.GPT2Decoder(configs.pretrained_model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model.to(device=self.device)

        resource.update(model=self.model)
        self.word_processor = tx.data.GPT2Tokenizer(
            pretrained_model_name=configs.pretrained_model_name)

        end_token = self.word_processor.map_token_to_id("<|endoftext|>")

        def _get_helper(start_tokens):
            if self.top_p:
                helper = tx.modules.TopPSampleEmbeddingHelper(
                    start_tokens=start_tokens,
                    end_token=end_token,
                    p=self.top_p,
                    softmax_temperature=self.temperature,
                )
            else:
                helper = tx.modules.TopKSampleEmbeddingHelper(
                    start_tokens=start_tokens,
                    end_token=end_token,
                    top_k=self.top_k,
                    softmax_temperature=self.temperature,
                )
            return helper

        self._get_helper = _get_helper

    @torch.no_grad()
    def predict(self, data_batch: Dict):

        preds: Dict = {
            # "srcSentId": data_batch["tid"],
            # We may use this field if we want to add links on the fly
            "input_sents_tids": [],
            "output_sents": [],
        }

        preds['input_sents_tids'] += data_batch['tid']

        context, context_length = self.get_batch_tensor(
            data_batch["context"], device=self.device)

        start_tokens = context[:, 0]
        max_decoding_length = self.max_decoding_length

        helper = self._get_helper(start_tokens)

        output, _ = self.model(
            context=context, context_sequence_length=context_length,
            max_decoding_length=max_decoding_length, helper=helper)

        sample_id = output.sample_id
        instance_num = len(sample_id)
        sentences = []
        complete_sentences = []
        for i in range(instance_num):
            si = sample_id[i][context_length[i]:]
            sentences.append(self.word_processor.map_id_to_text(si.tolist()))
            si = sample_id[i]
            complete_sentences.append(
                self.word_processor.map_id_to_text(si.tolist()))
        preds["output_sents"] += complete_sentences
        return preds

    def pack(self, data_pack: MultiPack, output_dict):
        """
        Write the prediction results back to datapack. If :attr:`_overwrite`
        is `True`, write the predicted ner to the original tokens.
        Otherwise, create a new set of tokens and write the predicted ner
        to the new tokens (usually use this configuration for evaluation.)
        """
        assert output_dict is not None
        output_pack = data_pack.get_pack(self.output_pack_name)

        input_sent_tids = output_dict["input_sents_tids"]
        output_sentences = output_dict["output_sents"]

        text = output_pack.text
        input_pack = data_pack.get_pack(self.input_pack_name)
        for input_id, output_sentence in zip(input_sent_tids, output_sentences):
            offset = len(output_pack.text)
            sent = Sentence(output_pack, offset, offset + len(output_sentence))
            output_pack.add_entry(sent)
            text += output_sentence + "\n"

            input_sent = input_pack.get_entry(input_id)
            cross_link = MultiPackLink(
                data_pack, data_pack.subentry(self.input_pack_name, input_sent),
                data_pack.subentry(self.output_pack_name, sent))
            data_pack.add_entry(cross_link)
            # We may also consider adding two link with opposite directions
            # Here the unidirectional link indicates the generation dependency
        output_pack.set_text(text)

    def get_batch_tensor(self, data: List, device):
        """

        Args:
            data: A list of strings(sentences)
            device:

        Returns:

        """
        batch_size = len(data)
        batch_tokens = [self.word_processor.map_text_to_token(sent)
                        for sent in data]

        batch_length = max([len(d) for d in batch_tokens])
        wid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)
        lengths = np.empty(batch_size, dtype=np.int64)

        for i, inst in enumerate(batch_tokens):
            wids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = \
                self.word_processor.map_token_to_id(wids)
            wid_inputs[i, inst_size:] = 0
            # The existence of length will mask these padding positions out
            # So we just set the padding value as 0,
            # which could be any in-range integers

        words = torch.from_numpy(wid_inputs).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        return words, lengths

    @staticmethod
    def default_hparams():
        return {
            'max_decoding_length': 128,
            'temperature': 0.7,
            'top_p': None,
            'top_k': 40,
            'pretrained_model_name': "117M",
            'checkpoint': None,
            'input_pack_name': None,
            'output_pack_name': None,
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            },
            'batch_size': 10,
        }
