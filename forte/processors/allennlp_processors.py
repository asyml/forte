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
from allennlp.predictors import Predictor
from texar.torch import HParams

from ft.onto.base_ontology import Token, Sentence, Dependency
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.common import ProcessorConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "AllenNLPProcessor",
]

# pylint: disable=line-too-long
MODEL2URL = {
    'stanford_dependencies': "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
    'universal_dependencies': "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz",
}
# pylint: enable=line-too-long


class AllenNLPProcessor(PackProcessor):

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: HParams):
        self.processors = configs.processors
        if self.processors is None or self.processors == "":
            self.processors = self.default_configs()['processors']

        if configs.output_format not in MODEL2URL:
            raise ProcessorConfigError('Incorrect value for output_format')
        model_url = MODEL2URL[configs.output_format]
        self.predictor = Predictor.from_path(model_url)

        self.overwrite_entries = configs.overwrite_entries
        self.allow_parallel_entries = configs.allow_parallel_entries
        if self.overwrite_entries:
            logger.warning("`overwrite_entries` is set to True, this means "
                           "that the entries of the same type as produced by "
                           "this processor will be overwritten if found.")
            if self.allow_parallel_entries:
                logger.warning('Both `overwrite_entries` (whether to overwrite '
                               'the entries of the same type as produced by '
                               'this processor) and '
                               '`allow_parallel_entries` (whether to allow '
                               'similar new entries when they already exist) '
                               'are True, all existing conflicting entries '
                               'will be deleted.')
        else:
            if not self.allow_parallel_entries:
                logger.warning('Both `overwrite_entries` (whether to overwrite '
                               'the entries of the same type as produced by '
                               'this processor) and '
                               '`allow_parallel_entries` (whether to allow '
                               'similar new entries when they already exist) '
                               'are False, processor will only run if there '
                               'are no existing conflicting entries.')

    @staticmethod
    def default_configs():
        """
        This defines a basic config structure for AllenNLP.
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - processors: defines what operations to be done on the sentence,
                default value is "tokenize,pos,depparse" which performs all the
                three operations
            - output_format: format of the POS tags and dependency parsing,
                default value is "universal_dependencies"
            - overwrite_entries: whether to overwrite the entries of the same
                type as produced by this processor, default value is False
            - allow_parallel_entries: whether to allow similar new entries when
                they already exist, e.g. allowing new tokens with same spans,
                used only when `overwrite_entries` is False
        """
        return {
            'processors': "tokenize,pos,depparse",
            'output_format': "universal_dependencies",
            'overwrite_entries': False,
            'allow_parallel_entries': True
        }

    def _process(self, input_pack: DataPack):
        # handle existing entries
        self._process_existing_entries(input_pack)

        for sentence in input_pack.get(Sentence):
            result = self.predictor.predict(sentence=sentence.text)

            if "tokenize" in self.processors:
                # creating new tokens and dependencies
                tokens = self._create_tokens(input_pack, sentence, result)
                if "depparse" in self.processors:
                    self._create_dependencies(input_pack, tokens, result)

    def _process_existing_entries(self, input_pack):
        tokens_exist = any(True for _ in input_pack.get(Token))
        dependencies_exist = any(True for _ in input_pack.get(Dependency))

        if tokens_exist or dependencies_exist:
            if not self.overwrite_entries:
                if not self.allow_parallel_entries:
                    raise ProcessorConfigError("Found existing entries, either "
                                               "`overwrite_entries` or "
                                               "`allow_parallel_entries` "
                                               "should be True")
            else:
                # delete existing tokens and dependencies
                for entry_type in (Token, Dependency):
                    for entry in input_pack.get(entry_type):
                        input_pack.delete_entry(entry)

    def _create_tokens(self, input_pack, sentence, result):
        words, pos = result['words'], result['pos']
        tokens = []
        offset = sentence.span.begin
        word_end = 0
        for i, word in enumerate(words):
            word_begin = sentence.text.find(word, word_end)
            word_end = word_begin + len(word)
            token = Token(input_pack,
                          offset + word_begin,
                          offset + word_end)
            if "pos" in self.processors:
                token.pos = pos[i]
            tokens.append(token)
            input_pack.add_entry(token)

        return tokens

    @staticmethod
    def _create_dependencies(input_pack, tokens, result):
        deps = result['predicted_dependencies']
        heads = result['predicted_heads']
        for i, token in enumerate(tokens):
            relation = Dependency(input_pack,
                                  parent=tokens[heads[i] - 1],
                                  child=token)
            relation.rel_type = deps[1]
            input_pack.add_or_get_entry(relation)
