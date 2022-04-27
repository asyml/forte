# Copyright 2021 The Forte Authors. All Rights Reserved.
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
"""Subword Tokenizer"""

__all__ = [
    "SubwordTokenizer",
]

from typing import List, Tuple, Iterator, Dict, Set


from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.base import PackProcessor
from forte.utils.utils import DiffAligner
from forte.utils import create_import_error_msg
from ft.onto.base_ontology import Subword


try:
    from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer
except ImportError as err1:
    raise ImportError(
        create_import_error_msg("texar-pytorch", "nlp", "NLP support")
    ) from err1

# This should probably be named as `BertTokenizer`.
class SubwordTokenizer(PackProcessor):
    """
    Subword Tokenizer using pretrained Bert model.
    """

    def __init__(self):
        super().__init__()

        self.tokenizer: BERTTokenizer = None
        self.aligner: DiffAligner = None
        self.__do_lower_case = True

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        if not self.configs.tokenizer_configs.pretrained_model_name:
            raise ValueError("Please specify a pretrained bert model")
        self.tokenizer = BERTTokenizer(
            cache_dir=None,
            hparams=self.configs.tokenizer_configs,
        )
        self.aligner = DiffAligner()
        self.__do_lower_case = self.configs.tokenizer_configs.do_lower_case

    def _process(self, input_pack: DataPack):
        assert self.tokenizer is not None
        assert self.aligner is not None

        if self.configs.token_source is not None:
            # Use provided token source.
            token: Annotation
            for token in input_pack.get(self.configs.token_source):
                assert isinstance(token, Annotation)
                self.__add_subwords(
                    input_pack,
                    token.text  # type: ignore
                    if self.__do_lower_case
                    else token.text,  # type: ignore
                    token.begin,  # type: ignore
                )
        elif self.configs.segment_unit is not None:
            # If token source not provide, try to use provided segments.
            segment: Annotation
            for segment in input_pack.get(self.configs.segment_unit):
                self._segment(
                    input_pack, segment.text, segment.begin  # type: ignore
                )
        else:
            # Use the whole data pack, maybe less efficient in some cases.
            self._segment(input_pack, input_pack.text, 0)

    def _word_tokenization(
        self, text: str
    ) -> Iterator[Tuple[str, Tuple[int, int]]]:
        """
        This function should tokenize the text and return the tokenization
        results in the form of a word and the span of each word. A span is the
        begin and end of this word, indexed from 0, and end = begin + length
        of the word.

        By default, this calls the Texar's BasicTokenizer and then align the
        result back. You can implement this function if you prefer a
        different tokenizer.

        Args:
            text: Input text to be tokenized.

        Returns:
            A iterator of tokenization result in the form of triplets of
            (word, (begin, end)).
        """
        basic_tokens: List[str] = self.tokenizer.basic_tokenizer.tokenize(
            text, never_split=self.tokenizer.all_special_tokens
        )
        token_spans = self.aligner.align_with_segments(text, basic_tokens)

        for t, span in zip(basic_tokens, token_spans):
            if span is not None:
                yield t, span

    def _segment(self, pack: DataPack, text: str, segment_offset: int):
        if self.tokenizer.do_basic_tokenize:
            for token, (token_begin, _) in self._word_tokenization(text):
                assert token is not None
                self.__add_subwords(pack, text, token_begin + segment_offset)
        else:
            self.__add_subwords(pack, text, segment_offset)

    def __add_subwords(self, pack: DataPack, text: str, text_offset: int):
        if self.__do_lower_case:
            lower_text = text.lower()
            # See this https://bugs.python.org/issue17252 to understand why this
            # is checked here. tl;dr sometimes lower casing special unicode
            # string will result in a change of length due to unicode NFD.
            if len(lower_text) == len(text):
                text = text.lower()

        for (
            subword,
            begin,
            end,
        ) in self.tokenizer.wordpiece_tokenizer.tokenize_with_span(text):
            subword_token = Subword(
                pack, begin + text_offset, end + text_offset
            )
            if subword == self.tokenizer.wordpiece_tokenizer.unk_token:
                subword_token.is_unk = True
            subword_token.is_first_segment = not subword.startswith("##")
            # pylint: disable=protected-access
            subword_token.vocab_id = self.tokenizer._map_token_to_id(subword)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the data pack storing type records needed
                in for consistency checking.

        Returns:
            None
        """
        record_meta["ft.onto.base_ontology.Subword"] = {
            "is_unk",
            "is_first_segment",
            "vocab_id",
        }

    def expected_types_and_attributes(self) -> Dict[str, Set[str]]:
        r"""Method to add expected type for current processor input which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        expected_types: Dict[str, Set[str]] = {}
        if self.configs.token_source is not None:
            expected_types[self.configs.token_source] = set()
        elif self.configs.segment_unit is not None:
            expected_types[self.configs.segment_unit] = set()
        return expected_types

    @classmethod
    def default_configs(cls):
        """Returns the configuration with default values.

        Here:

        - `tokenizer_configs` contains all default hyper-parameters in
          :class:`~texar.torch.data.tokenizer.bert_tokenizer.BERTTokenizer`,
          this processor will pass on all the configurations to the
          tokenizer to create the tokenizer instance.

        - `segment_unit` contains an Annotation entry type used to split the
          text into smaller units. For example, setting this to
          `ft.onto.base_ontology.Sentence` will make this tokenizer do
          tokenization on a sentence base, which could be more efficient
          when the alignment is used.

        - `token_source` contains entry name of where the tokens come from.
          For example, setting this to `ft.onto.base_ontology.Token` will
          make this tokenizer split the sub-word based on this token. The
          default value will use `ft.onto.base_ontology.Token`. If this
          value is set to None, then it will use `word_tokenization`
          function of this class to do tokenization.

        Note that if `segment_unit` or `token_source` is provided, the
        :meth:`~forte.processors.base.base_processor.BaseProcessor.check_record`
        will check if certain types are written before this processor.

        Returns: Default configuration value for the tokenizer.

        """
        return {
            "tokenizer_configs": BERTTokenizer.default_hparams(),
            "segment_unit": None,
            "token_source": "ft.onto.base_ontology.Token",
            "@no_typecheck": "token_source",
        }
