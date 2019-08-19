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
from forte.data.ontology import base_ontology
from forte.models.gpt import processor
from forte.processors.base.batch_processor import \
    MultiPackTxtgenBatchProcessor

logger = logging.getLogger(__name__)


class TxtgenPredictor(MultiPackTxtgenBatchProcessor):
    def __init__(self):
        super().__init__()

        self.word_processor = None
        self.model = None
        self.ontology = base_ontology

        self.batch_size = 6
        self._get_helper = None

        self.current_datapack: MultiPack = None
        self.max_decoding_length = None
        self.temperature = None
        self.top_k = None
        self.top_p = None
        self.device = None
        self.define_input_info()
        self.define_output_info()
        self.define_context()

    def define_input_info(self) -> None:
        """
        Define the input info for each Data pack in the MultiPack
        for future query
        """
        self.input_info = {
            self.ontology.Sentence: []
        }

    def define_output_info(self) -> None:
        self.output_info = {
            self.ontology.Sentence: []
        }

    def define_context(self):
        self.context_type = self._ontology.Sentence

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

        self.input_pack_name = configs.input_pack_name

        # pylint: disable=attribute-defined-outside-init
        self.batcher = self.initialize_batcher()

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
        self.define_input_info()
        self.define_output_info()

    def process(self, input_pack: MultiPack, tail_instances: bool = False):
        """
        :param input_pack: A MultiPack instance
        :param tail_instances:
        :return:
        """
        self.current_datapack = input_pack
        config.working_component = self.component_name

        # Read data from the "Input_pack" of the input
        for batch in self.batcher.get_batch(
                self.current_datapack,
                self.context_type,
                self.input_info,
                tail_instances=tail_instances,
        ):
            print('current_batch_size:{}'.format(len(batch['context'])))
            pred = self.predict(batch)
            self.pack_all(pred)
            self.finish_up_packs(-1)
        if len(self.batcher.current_batch_sources) == 0:
            self.finish_up_packs()

        config.working_component = None

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
        output_pack = data_pack.packs[self.output_pack_name]

        input_sent_tids = output_dict["input_sents_tids"]
        output_sentences = output_dict["output_sents"]

        text = output_pack.text
        input_pack = data_pack.packs[self.input_pack_name]
        for input_id, output_sentence in zip(input_sent_tids, output_sentences):
            offset = len(output_pack.text)
            sent = self.ontology.Sentence(offset, offset + len(output_sentence))
            output_pack.add_entry(sent)
            text += output_sentence + "\n"

            input_sent = input_pack.get_entry_by_id(input_id)
            cross_link = MultiPackLink(input_sent,
                                       sent)
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
