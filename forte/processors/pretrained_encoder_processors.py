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

import torch

import texar.torch as tx

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.ontology.top import Annotation
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.utils.utils import get_class


__all__ = [
    "PretrainedEncoder",
]


class PretrainedEncoder(PackProcessor):
    r"""A wrapper of Texar pre-trained encoders.

    This processor will compute the embedding vectors of the corresponding
    annotations (determined by the user) by using the pre-trained encoders.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.encoder = None
        self.entry_type = None

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources, configs: HParams):
        if configs.pretrained_model_name.startswith('bert') or \
                configs.pretrained_model_name.startswith('biobert') or \
                configs.pretrained_model_name.startswith('scibert'):
            self.tokenizer = tx.data.BERTTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.BERTEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('gpt2'):
            self.tokenizer = tx.data.GPT2Tokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.GPT2Encoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('roberta'):
            self.tokenizer = tx.data.RoBERTaTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.RoBERTaEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('T5'):
            self.tokenizer = tx.data.T5Tokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.T5Encoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('xlnet'):
            self.tokenizer = tx.data.XLNetTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.XLNetEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        else:
            raise ValueError("Unrecognized pre-trained model name.")

        self.entry_type = get_class(configs.entry_type)
        if not isinstance(self.entry_type, Annotation) and \
                not issubclass(self.entry_type, Annotation):
            raise ValueError("entry_type must be annotation type.")

    def _process(self, input_pack: DataPack):
        for entry in input_pack.get(entry_type=self.entry_type):
            input_ids, segment_ids, _ = self.tokenizer.encode_text(
                text_a=entry.text)

            input_ids = torch.tensor([input_ids])
            segment_ids = torch.tensor([segment_ids])
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            output, _ = self.encoder(input_ids, input_length, segment_ids)
            entry.embedding = output.tolist()

    @staticmethod
    def default_configs():
        r"""This default configurations for :class:`PretrainedEncoder`.

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained models including:

            * **Standard BERT**: proposed in (`Devlin et al`. 2018)
              `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
              . A bidirectional Transformer language model pre-trained on large
              text corpora. Available model names include:

                * ``bert-base-uncased``: 12-layer, 768-hidden, 12-heads,
                  110M parameters.
                * ``bert-large-uncased``: 24-layer, 1024-hidden, 16-heads,
                  340M parameters.
                * ``bert-base-cased``: 12-layer, 768-hidden, 12-heads ,
                  110M parameters.
                * ``bert-large-cased``: 24-layer, 1024-hidden, 16-heads,
                  340M parameters.
                * ``bert-base-multilingual-uncased``: 102 languages, 12-layer,
                  768-hidden, 12-heads, 110M parameters.
                * ``bert-base-multilingual-cased``: 104 languages, 12-layer,
                  768-hidden, 12-heads, 110M parameters.
                * ``bert-base-chinese``: Chinese Simplified and Traditional,
                  12-layer, 768-hidden, 12-heads, 110M parameters.

            * **BioBERT**: proposed in (`Lee et al`. 2019)
              `BioBERT: a pre-trained biomedical language representation model for biomedical text mining`_
              . A domain specific language representation model pre-trained on
              large-scale biomedical corpora. Based on the BERT architecture,
              BioBERT effectively transfers the knowledge from a large amount
              of biomedical texts to biomedical text mining models with minimal
              task-specific architecture modifications. Available model names
              include:

              * ``biobert-v1.0-pmc``: BioBERT v1.0 (+ PMC 270K) - based on
                BERT-base-Cased (same vocabulary).
              * ``biobert-v1.0-pubmed-pmc``: BioBERT v1.0 (+ PubMed 200K + PMC
                270K) - based on BERT-base-Cased (same vocabulary).
              * ``biobert-v1.0-pubmed``: BioBERT v1.0 (+ PubMed 200K) - based on
                BERT-base-Cased (same vocabulary).
              * ``biobert-v1.1-pubmed``: BioBERT v1.1 (+ PubMed 1M) - based on
                BERT-base-Cased (same vocabulary).

            * **SciBERT**: proposed in (`Beltagy et al`. 2019)
              `SciBERT: A Pretrained Language Model for Scientific Text`_. A
              BERT model trained on scientific text. SciBERT leverages
              unsupervised pre-training on a large multi-domain corpus of
              scientific publications to improve performance on downstream
              scientific NLP tasks. Available model names include:

              * ``scibert-scivocab-uncased``: Uncased version of the model
                trained on its own vocabulary.
              * ``scibert-scivocab-cased``: Cased version of the model trained
                on its own vocabulary.
              * ``scibert-basevocab-uncased``: Uncased version of the model
                trained on the original BERT vocabulary.
              * ``scibert-basevocab-cased``: Cased version of the model trained
                on the original BERT vocabulary.

            * The GPT2 model was proposed in
              `Language Models are Unsupervised Multitask Learners`_
              by `Radford et al.` from OpenAI. It is a unidirectional
              Transformer model pre-trained using the vanilla language modeling
              objective on a large corpus.

              The available GPT2 models are as follows:

              * ``gpt2-small``: Small version of GPT-2, 124M parameters.
              * ``gpt2-medium``: Medium version of GPT-2, 355M parameters.
              * ``gpt2-large``: Large version of GPT-2, 774M parameters.
              * ``gpt2-xl``: XL version of GPT-2, 1558M parameters.

            * The RoBERTa model was proposed in (`Liu et al`. 2019)
              `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_.
              As a variant of the standard BERT model, RoBERTa trains for more
              iterations on more data with a larger batch size as well as other
              tweaks in pre-training. Differing from the standard BERT, the
              RoBERTa model does not use segmentation embedding. Available
              model names include:

              * ``roberta-base``: RoBERTa using the BERT-base architecture,
                125M parameters.
              * ``roberta-large``: RoBERTa using the BERT-large architecture,
                355M parameters.

            * The T5 model treats multiple NLP tasks in a similar manner by
              encoding the different tasks as text directives in the input
              stream. This enables a single model to be trained supervised on a
              wide variety of NLP tasks.

              The T5 model examines factors relevant for leveraging transfer
              learning at scale from pure unsupervised pre-training to
              supervised tasks. It is discussed in much detail in
              `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
              from Google.

              The available T5 models are as follows:

              * ``T5-Small``: Small version of T5, 60 million parameters.
              * ``T5-Base``: Base-line version of T5, 220 million parameters.
              * ``T5-Large``: Large Version of T5, 770 million parameters.
              * ``T5-3B``: A version of T5 with 3 billion parameters.
              * ``T5-11B``: A version of T5 with 11 billion parameters.

            * The XLNet model was proposed in
              `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
              by `Yang et al.` It is based on the Transformer-XL model,
              pre-trained on a large corpus using a language modeling objective
              that considers all permutations of the input sentence.

              The available XLNet models are as follows:

              * ``xlnet-based-cased``: 12-layer, 768-hidden, 12-heads. This
                model is trained on full data (different from the one in the
                paper).
              * ``xlnet-large-cased``: 24-layer, 1024-hidden, 16-heads.

        `"entry_type"`: str
            The annotation type that user want to generate embbeding on .

        .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
            https://arxiv.org/abs/1810.04805

        .. _`BioBERT: a pre-trained biomedical language representation model for biomedical text mining`:
            https://arxiv.org/abs/1901.08746

        .. _`SciBERT: A Pretrained Language Model for Scientific Text`:
            https://arxiv.org/abs/1903.10676

        .. _`Language Models are Unsupervised Multitask Learners`:
            https://openai.com/blog/better-language-models/

        .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
            https://arxiv.org/abs/1907.11692

        .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
            https://arxiv.org/abs/1910.10683

        .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
            http://arxiv.org/abs/1906.08237
        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'entry_type': 'ft.onto.base_ontology.Sentence',
        }
