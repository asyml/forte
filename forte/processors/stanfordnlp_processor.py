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

from typing import List, Any, Dict

import stanza
from texar.torch import HParams

from ft.onto.base_ontology import Token, Sentence, Dependency
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "StandfordNLPProcessor",
]


class StandfordNLPProcessor(PackProcessor):
    def __init__(self, models_path: str):
        super().__init__()
        self.processors = ""
        self.nlp = None
        self.MODELS_DIR = models_path
        self.lang = 'en'  # English is default

    def set_up(self):
        stanza.download(self.lang, self.MODELS_DIR)

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: HParams):
        self.processors = configs.processors
        self.lang = configs.lang
        self.set_up()
        self.nlp = stanza.Pipeline(**configs.todict(),
                                   models_dir=self.MODELS_DIR)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        This defines a basic config structure for StanfordNLP.
        :return:
        """
        config = super().default_configs()
        config.update(
            {
                'processors': 'tokenize,pos,lemma,depparse',
                'lang': 'en',
                # Language code for the language to build the Pipeline
                'use_gpu': False,
            })
        return config

    def _process(self, input_pack: DataPack):
        doc = input_pack.text
        end_pos = 0

        # sentence parsing
        sentences = self.nlp(doc).sentences  # type: ignore

        # Iterating through stanfordnlp sentence objects
        for sentence in sentences:
            begin_pos = doc.find(sentence.words[0].text, end_pos)
            end_pos = doc.find(sentence.words[-1].text, begin_pos) + len(
                sentence.words[-1].text)
            sentence_entry = Sentence(input_pack, begin_pos, end_pos)
            input_pack.add_or_get_entry(sentence_entry)

            tokens: List[Token] = []
            if "tokenize" in self.processors:
                offset = sentence_entry.span.begin
                end_pos_word = 0

                # Iterating through stanfordnlp word objects
                for word in sentence.words:
                    begin_pos_word = sentence_entry.text. \
                        find(word.text, end_pos_word)
                    end_pos_word = begin_pos_word + len(word.text)
                    token = Token(input_pack,
                                  begin_pos_word + offset,
                                  end_pos_word + offset
                                  )

                    if "pos" in self.processors:
                        token.pos = word.pos
                        token.ud_xpos = word.xpos

                    if "lemma" in self.processors:
                        token.lemma = word.lemma

                    token = input_pack.add_or_get_entry(token)
                    tokens.append(token)

            # For each sentence, get the dependency relations among tokens
            if "depparse" in self.processors:
                # Iterating through token entries in current sentence
                for token, word in zip(tokens, sentence.words):
                    child = token  # current token
                    parent = tokens[word.governor - 1]  # Root token
                    relation_entry = Dependency(input_pack, parent, child)
                    relation_entry.rel_type = word.dependency_relation

                    input_pack.add_or_get_entry(relation_entry)
