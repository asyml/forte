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

import os
import spacy

from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Token, Sentence


class SpacySentenceSegmenter(PackProcessor):
    """
    A wrapper of spaCy sentence segmenter.
    """

    def _process(self, input_pack: DataPack):
        text = input_pack.text
        end_pos = 0
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli.download import download
            download('en_core_web_sm')
            nlp = spacy.load("en_core_web_sm")

        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            doc = nlp(paragraph)
            for sent in doc.sents:
                begin_pos = text.find(sent, end_pos)
                end_pos = begin_pos + len(sent)
                sentence_entry = Sentence(input_pack, begin_pos, end_pos)
                input_pack.add_or_get_entry(sentence_entry)


class SpacyWordTokenizer(PackProcessor):
    """
    A wrapper for spaCy word tokenizer
    """
    def __init__(self):
        super().__init__()
        self.sentence_component = None

    def _process(self, input_pack: DataPack):

        for sentence in input_pack.get(entry_type=Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin

            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli.download import download
                download('en_core_web_sm')
                nlp = spacy.load("en_core_web_sm")

            doc = nlp(sentence.text)
            for word in doc:
                begin_pos = sentence.text.find(word, end_pos)
                end_pos = begin_pos + len(word)
                token = Token(input_pack, begin_pos + offset, end_pos + offset)
                input_pack.add_or_get_entry(token)


class SpacyTokenizer(PackProcessor):
    """
    A wrapper for spaCy POSTagger
    """
    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(input_pack.get(entry_type=Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli.download import download
                download('en_core_web_sm')
                nlp = spacy.load("en_core_web_sm")

            token_texts = [token.text for token in token_entries]
            taggings = pos_tag(token_texts)
            for token, tag in zip(token_entries, taggings):
                token.set_fields(pos=tag[1])
