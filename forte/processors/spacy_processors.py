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

from typing import List

import spacy
from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Token, Sentence


__all__ = [
    "SpacyProcessor",
]


class SpacyProcessor(PackProcessor):
    """
    A wrapper for spaCy processors
    """
    def __init__(self):
        super().__init__()
        self.processors = ""
        self.nlp = None
        self.lang = 'en_core_web_sm'

    def set_up(self):
        try:
            self.nlp = spacy.load(self.lang)
        except OSError:
            from spacy.cli.download import download
            download(self.lang)
            self.nlp = spacy.load(self.lang)

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources,
                   configs: HParams):
        self.processors = configs.processors
        self.lang = configs.lang
        self.set_up()

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure for spaCy.
        Returns:

        """
        return {
            'processors': 'tokenize, pos, lemma',
            'lang': 'en_core_web_sm',
            # Language code for the language to build the Pipeline
            'use_gpu': False,
        }

    def _process(self, input_pack: DataPack):
        doc = input_pack.text
        end_pos = 0

        # sentence parsing
        sentences = self.nlp(doc).sents

        for sentence in sentences:
            begin_pos = doc.find(sentence[0].text, end_pos)
            end_pos = doc.find(sentence[-1].text, begin_pos) + len(
                sentence[-1].text)
            sentence_entry = Sentence(input_pack, begin_pos, end_pos)
            input_pack.add_or_get_entry(sentence_entry)

            tokens: List[Token] = []

            if "tokenize" in self.processors:
                offset = sentence_entry.span.begin
                end_pos_word = 0

                # Iterating through spaCy token objects
                for word in sentence:
                    begin_pos_word = sentence_entry.text.find(word.text,
                                                              end_pos_word)
                    end_pos_word = begin_pos_word + len(word.text)
                    token = Token(input_pack, begin_pos_word + offset,
                                  end_pos_word + offset)

                    if "pos" in self.processors:
                        token.set_fields(pos=word.tag_)

                    if "lemma" in self.processors:
                        token.set_fields(lemma=word.lemma_)

                    tokens.append(token)
                    input_pack.add_or_get_entry(token)
