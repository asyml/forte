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

import spacy
from spacy.language import Language
from texar.torch import HParams

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
        self.nlp: Language = None
        self.lang_model: str = ''

    def set_up(self):
        try:
            self.nlp = spacy.load(self.lang_model)
        except OSError:
            from spacy.cli.download import download
            download(self.lang_model)
            self.nlp = spacy.load(self.lang_model)

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources,
                   configs: HParams):
        self.processors = configs.processors
        self.lang_model = configs.lang
        self.set_up()

    @staticmethod
    def default_configs():
        """
        This defines a basic config structure for spaCy.
        Returns:

        """
        return {
            'processors': 'tokenize, pos, lemma',
            'lang': 'en_core_web_sm',
            # Language code for the language to build the Pipeline
            'use_gpu': False,
        }

    def _process_parser(self, sentences, input_pack):
        """Parse the sentence. Default behaviour is to segment sentence, POSTag
        and Lemmatize.

        Args:
            sentences: Generator object which yields sentences in document
            input_pack: input pack which needs to be modified

        Returns:

        """
        for sentence in sentences:
            sentence_entry = Sentence(input_pack,
                                      sentence.start_char,
                                      sentence.end_char)
            input_pack.add_or_get_entry(sentence_entry)

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

                    input_pack.add_or_get_entry(token)

    def _process_ner(self, doc, input_pack):
        """Perform spaCy's NER Pipeline on the document.

        Args:
            doc: The document
            input_pack: Input pack to fill

        Returns:

        """
        ner_doc = self.nlp(doc)

        for item in ner_doc.ents:
            entity = EntityMention(input_pack, item.start_char,
                                   item.end_char)
            entity.ner_type = item.label_
            input_pack.add_or_get_entry(entity)

    def _process(self, input_pack: DataPack):
        doc = input_pack.text

        if 'ner' in self.processors:
            self._process_ner(doc, input_pack)

        try:
            # sentence parsing
            sentences = self.nlp(doc).sents
            self._process_parser(sentences, input_pack)
        except ValueError:
            raise ValueError(f"The provided language model does not support"
                             f" SpaCy's parser pipeline. Refer to "
                             f"https://spacy.io/models/ for more information."
                             f" Please check input and try again.")
