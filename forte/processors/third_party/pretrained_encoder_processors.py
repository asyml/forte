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


from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.processors.base import PackProcessor
from forte.utils.utils import get_class

__all__ = [
    "PretrainedEncoder",
]


class PretrainedEncoder(PackProcessor):
    r"""A wrapper of Texar pre-trained encoders.

    This processor will compute the embedding vectors for entries of type
    :class:`~forte.data.ontology.top.Annotation` using pre-trained models. The user can specify the
    pre-trained model type and the annotation class name via configuration.
    For the full list of pre-trained models supported, see
    :meth:`default_config` for more details. The processor will add embedding
    vector for all entries matching the specified entry type. The resulting
    vector can be accessed by the embedding field of the annotations.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.encoder = None
        self.entry_type = None
        try:
            import texar.torch as tx  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                " `texar-pytorch` is not installed correctly."
                " Please refer to [extra requirement for nlp](pip"
                " install forte[nlp])"
                " for more information. "
            ) from err
        self.name2tokenizer = {
            "bert-base-uncased": tx.data.BERTTokenizer,
            "bert-large-uncased": tx.data.BERTTokenizer,
            "bert-base-cased": tx.data.BERTTokenizer,
            "bert-large-cased": tx.data.BERTTokenizer,
            "bert-base-multilingual-uncased": tx.data.BERTTokenizer,
            "bert-base-multilingual-cased": tx.data.BERTTokenizer,
            "bert-base-chinese": tx.data.BERTTokenizer,
            "biobert-v1.0-pmc": tx.data.BERTTokenizer,
            "biobert-v1.0-pubmed-pmc": tx.data.BERTTokenizer,
            "biobert-v1.0-pubmed": tx.data.BERTTokenizer,
            "biobert-v1.1-pubmed": tx.data.BERTTokenizer,
            "scibert-scivocab-uncased": tx.data.BERTTokenizer,
            "scibert-scivocab-cased": tx.data.BERTTokenizer,
            "scibert-basevocab-uncased": tx.data.BERTTokenizer,
            "scibert-basevocab-cased": tx.data.BERTTokenizer,
            "gpt2-small": tx.data.GPT2Tokenizer,
            "gpt2-medium": tx.data.GPT2Tokenizer,
            "gpt2-large": tx.data.GPT2Tokenizer,
            "gpt2-xl": tx.data.GPT2Tokenizer,
            "roberta-base": tx.data.RoBERTaTokenizer,
            "roberta-large": tx.data.RoBERTaTokenizer,
            "T5-Small": tx.data.T5Tokenizer,
            "T5-Base": tx.data.T5Tokenizer,
            "T5-Large": tx.data.T5Tokenizer,
            "T5-3B": tx.data.T5Tokenizer,
            "T5-11B": tx.data.T5Tokenizer,
            "xlnet-based-cased": tx.data.XLNetTokenizer,
            "xlnet-large-cased": tx.data.XLNetTokenizer,
        }
        self.name2encoder = {
            "bert-base-uncased": tx.modules.BERTEncoder,
            "bert-large-uncased": tx.modules.BERTEncoder,
            "bert-base-cased": tx.modules.BERTEncoder,
            "bert-large-cased": tx.modules.BERTEncoder,
            "bert-base-multilingual-uncased": tx.modules.BERTEncoder,
            "bert-base-multilingual-cased": tx.modules.BERTEncoder,
            "bert-base-chinese": tx.modules.BERTEncoder,
            "biobert-v1.0-pmc": tx.modules.BERTEncoder,
            "biobert-v1.0-pubmed-pmc": tx.modules.BERTEncoder,
            "biobert-v1.0-pubmed": tx.modules.BERTEncoder,
            "biobert-v1.1-pubmed": tx.modules.BERTEncoder,
            "scibert-scivocab-uncased": tx.modules.BERTEncoder,
            "scibert-scivocab-cased": tx.modules.BERTEncoder,
            "scibert-basevocab-uncased": tx.modules.BERTEncoder,
            "scibert-basevocab-cased": tx.modules.BERTEncoder,
            "gpt2-small": tx.modules.GPT2Encoder,
            "gpt2-medium": tx.modules.GPT2Encoder,
            "gpt2-large": tx.modules.GPT2Encoder,
            "gpt2-xl": tx.modules.GPT2Encoder,
            "roberta-base": tx.modules.RoBERTaEncoder,
            "roberta-large": tx.modules.RoBERTaEncoder,
            "T5-Small": tx.modules.T5Encoder,
            "T5-Base": tx.modules.T5Encoder,
            "T5-Large": tx.modules.T5Encoder,
            "T5-3B": tx.modules.T5Encoder,
            "T5-11B": tx.modules.T5Encoder,
            "xlnet-based-cased": tx.modules.XLNetEncoder,
            "xlnet-large-cased": tx.modules.XLNetEncoder,
        }

    def available_checkpoints(self):
        return list(self.name2tokenizer.keys())

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        if configs.pretrained_model_name in self.name2tokenizer:
            self.tokenizer = self.name2tokenizer[configs.pretrained_model_name](
                pretrained_model_name=configs.pretrained_model_name
            )
            self.encoder = self.name2encoder[configs.pretrained_model_name](
                pretrained_model_name=configs.pretrained_model_name
            )
        else:
            raise ValueError("Unrecognized pre-trained model name.")

        self.entry_type = get_class(configs.entry_type)
        if not isinstance(self.entry_type, Annotation) and not issubclass(
            self.entry_type, Annotation
        ):
            raise ValueError("entry_type must be annotation type.")

    def _process(self, input_pack: DataPack):
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                " `texar-pytorch` is not installed correctly."
                " Please refer to [extra requirement for nlp]"
                "(pip install forte[nlp])"
                " for more information. "
            ) from err
        for entry in input_pack.get(entry_type=self.entry_type):  # type: ignore
            input_ids, segment_ids, _ = self.tokenizer.encode_text(
                text_a=entry.text
            )

            input_ids = torch.tensor([input_ids])
            segment_ids = torch.tensor([segment_ids])
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            output, _ = self.encoder(input_ids, input_length, segment_ids)
            entry.embedding = output.tolist()

    @classmethod
    def default_configs(cls):
        r"""This default configurations for :class:`PretrainedEncoder`.

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained models including:

            * **Standard BERT**: proposed in (`Devlin et al`. 2018)
              `BERT: Pre-training of Deep Bidirectional Transformers for
              Language Understanding`_
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
              `BioBERT: a pre-trained biomedical language representation model
              for biomedical text mining`_
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
              `Exploring the Limits of Transfer Learning with a Unified
              Text-to-Text Transformer`_
              from Google.

              The available T5 models are as follows:

              * ``T5-Small``: Small version of T5, 60 million parameters.
              * ``T5-Base``: Base-line version of T5, 220 million parameters.
              * ``T5-Large``: Large Version of T5, 770 million parameters.
              * ``T5-3B``: A version of T5 with 3 billion parameters.
              * ``T5-11B``: A version of T5 with 11 billion parameters.

            * The XLNet model was proposed in
              `XLNet: Generalized Autoregressive Pretraining for Language
              Understanding`_
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

        .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language
         Understanding`:
            https://arxiv.org/abs/1810.04805

        .. _`BioBERT: a pre-trained biomedical language representation model
        for biomedical text mining`:
            https://arxiv.org/abs/1901.08746

        .. _`SciBERT: A Pretrained Language Model for Scientific Text`:
            https://arxiv.org/abs/1903.10676

        .. _`Language Models are Unsupervised Multitask Learners`:
            https://openai.com/blog/better-language-models/

        .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
            https://arxiv.org/abs/1907.11692

        .. _`Exploring the Limits of Transfer Learning with a Unified
        Text-to-Text Transformer`:
            https://arxiv.org/abs/1910.10683

        .. _`XLNet: Generalized Autoregressive Pretraining for Language
        Understanding`:
            http://arxiv.org/abs/1906.08237
        """

        return {
            "pretrained_model_name": "bert-base-uncased",
            "entry_type": "ft.onto.base_ontology.Sentence",
        }
