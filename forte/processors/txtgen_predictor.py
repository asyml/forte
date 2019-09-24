import logging
from typing import Dict, List

import numpy as np
import torch
import texar.torch as tx
from texar.torch.hyperparams import HParams

from forte import config
from forte.common.resources import Resources
from forte.data import MultiPack
from forte.data import MultiPackLink
from forte.data.batchers import ProcessingBatcher, \
    FixedSizeMultiPackProcessingBatcher
from forte.data.ontology import base_ontology
from forte.models.gpt import processor
from forte.processors.base.batch_processor import \
    MultiPackBatchProcessor
from forte.processors.base import ProcessInfo

logger = logging.getLogger(__name__)


class TxtgenPredictor(MultiPackBatchProcessor):

    def __init__(self):
        super().__init__()
        self.input_pack_name = None
        self.output_pack_name = None

        self.word_processor = None
        self.model = None
        self.ontology = base_ontology

        self.batch_size = 6
        self._get_helper = None

        self.max_decoding_length = None
        self.temperature = None
        self.top_k = None
        self.top_p = None
        self.device = None
        self.define_context()

    def _define_input_info(self) -> ProcessInfo:
        """
        Define the input info for each Data pack in the MultiPack
        for future query
        """
        input_info: ProcessInfo = {
            self.ontology.Sentence: []
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self.ontology.Sentence: []
        }
        return output_info

    def define_context(self):
        self.context_type = self._ontology.Sentence

    def define_batcher(self) -> ProcessingBatcher:
        return FixedSizeMultiPackProcessingBatcher()

    def initialize(self, configs: HParams, resource: Resources):
        """
        :param configs:

        :param resource:
            word_processor: encode the plain sentence with customized tokenizer
            input_pack_name: specify the input pack name of the MultiPack to be
             processed
            output_pack_name: specify the output pack name of the MultiPack to
             be processed
        :return:
        """
        super().initialize(configs, resource)

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
        self.word_processor = processor.get_encoder(
            self.model.pretrained_model_dir)

        end_token = self.word_processor.encoder["<|endoftext|>"]

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
        self._define_input_info()
        self._define_output_info()

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
            data_batch["context"], device=self.device
        )
        start_tokens = context[:, 0]
        max_decoding_length = self.max_decoding_length

        helper = self._get_helper(start_tokens)

        output, _ = self.model(
            context=context,
            context_sequence_length=context_length,
            max_decoding_length=max_decoding_length,
            helper=helper,
        )

        sample_id = output.sample_id
        instance_num = len(sample_id)
        sentences = []
        complete_sentences = []
        for i in range(instance_num):
            si = sample_id[i][context_length[i]:]
            sentences.append(self.word_processor.decode(si.tolist()))
            si = sample_id[i]
            complete_sentences.append(self.word_processor.decode(si.tolist()))
        preds["output_sents"] += complete_sentences
        return preds

    def pack(self, data_pack: MultiPack, output_dict):
        """
        Write the prediction results back to datapack. If :attr:`_overwrite`
        is `True`, write the predicted ner_tag to the original tokens.
        Otherwise, create a new set of tokens and write the predicted ner_tag
        to the new tokens (usually use this configuration for evaluation.)
        """
        assert output_dict is not None
        output_pack = data_pack._packs[self.output_pack_name]

        input_sent_tids = output_dict["input_sents_tids"]
        output_sentences = output_dict["output_sents"]

        text = output_pack.text
        input_pack = data_pack._packs[self.input_pack_name]
        for input_id, output_sentence in zip(input_sent_tids, output_sentences):
            offset = len(output_pack.text)
            sent = self.ontology.Sentence(
                output_pack, offset, offset + len(output_sentence)
            )
            output_pack.add_entry(sent)
            text += output_sentence + "\n"

            input_sent = input_pack.get_entry_by_id(input_id)
            cross_link = MultiPackLink(
                data_pack,
                (self.input_pack_name, input_sent),
                (self.output_pack_name, sent),
            )
            data_pack.add_entry(cross_link)
            # We may also consider adding two link with opposite directions
            # Here the unidirectional link indicates the generation dependency
        output_pack.set_text(text)

    def _record_fields(self, data_pack: MultiPack):
        data_pack.record_fields(
            ["span"], self.ontology.Sentence, self.component_name
        )

    def get_batch_tensor(self, data: List, device):
        """
        :param data: A list of strings(sentences)
        :param device:
        :return:
        """
        batch_size = len(data)
        batch_tokens = [self.word_processor.encode(sent) for sent in data]

        batch_length = max([len(d) for d in batch_tokens])
        wid_inputs = np.empty([batch_size, batch_length], dtype=np.int64)
        lengths = np.empty(batch_size, dtype=np.int64)

        for i, inst in enumerate(batch_tokens):
            wids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
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
                'type': 'nlp.forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
