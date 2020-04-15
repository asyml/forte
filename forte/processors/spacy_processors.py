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
from typing import Optional

import spacy
from spacy.language import Language

from forte.common import ProcessExecutionException
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import EntityMention, Sentence, Token

__all__ = [
    "SpacyProcessor",
]


class SpacyProcessor(PackProcessor):
    """
    A wrapper for spaCy processors
    """

    def __init__(self):
        super().__init__()
        self.processors: str = ""
        self.nlp: Optional[Language] = None
        self.lang_model: str = ''

    def set_up(self):
        try:
            self.nlp = spacy.load(self.lang_model)
        except OSError:
            from spacy.cli.download import download
            download(self.lang_model)
            self.nlp = spacy.load(self.lang_model)

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        self.processors = configs.processors
        self.lang_model = configs.lang
        self.set_up()

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for spaCy.
        Returns:

        """
        config = super().default_configs()
        config.update({
            'processors': 'tokenize, pos, lemma',
            'lang': 'en_core_web_sm',
            # Language code for the language to build the Pipeline
            'use_gpu': False,
        })
        return config

    def _process_parser(self, sentences, input_pack):
        """Parse the sentence. Default behaviour is to segment sentence, POSTag
        and Lemmatize.

        Args:
            sentences: Generator object which yields sentences in document
            input_pack: input pack which needs to be modified

        Returns:

        """
        for sentence in sentences:
            Sentence(input_pack, sentence.start_char, sentence.end_char)

            if "tokenize" in self.processors:
                # Iterating through spaCy token objects
                for word in sentence:
                    begin_pos_word = word.idx
                    end_pos_word = begin_pos_word + len(word.text)
                    token = Token(input_pack, begin_pos_word, end_pos_word)

                    if "pos" in self.processors:
                        token.pos = word.tag_

                    if "lemma" in self.processors:
                        token.lemma = word.lemma_

    def _process_ner(self, result, input_pack):
        """Perform spaCy's NER Pipeline on the document.

        Args:
            result: SpaCy results
            input_pack: Input pack to fill

        Returns:

        """
        for item in result.ents:
            entity = EntityMention(input_pack, item.start_char,
                                   item.end_char)
            entity.ner_type = item.label_

    def _process(self, input_pack: DataPack):
        doc = input_pack.text

        # Do all process.
        if self.nlp is None:
            raise ProcessExecutionException(
                "The SpaCy pipeline is not initialized, maybe you "
                "haven't called the initialization function.")
        result = self.nlp(doc)

        # Record NER results.
        self._process_ner(result, input_pack)

        # Process sentence parses.
        self._process_parser(result.sents, input_pack)
