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
"""
Unit tests for spaCy processors.
"""
import unittest
from typing import List

from ddt import ddt, data
import spacy
from spacy.language import Language

from forte.common import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.spacy_processors import SpacyProcessor
from ft.onto.base_ontology import Token, EntityMention


@ddt
class TestSpacyProcessor(unittest.TestCase):
    def setUp(self):
        self.spacy = Pipeline[DataPack]()
        self.spacy.set_reader(StringReader())

        config = {
            "processors": "tokenize",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }
        self.spacy.add(SpacyProcessor(), config=config)
        self.spacy.initialize()

        self.nlp: Language = spacy.load(config['lang'])

    def test_spacy_processor(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.spacy.process(document)

        # Check document
        self.assertEqual(pack.text, document)

        # Check tokens
        tokens = [x.text for x in pack.annotations if isinstance(x, Token)]
        document = document.replace('.', ' .')
        self.assertEqual(tokens, document.split())

    @data(
        "tokenize",
        "tokenize, pos",
        "tokenize, pos, lemma",
        "tokenize, lemma",
        "lemma",
        "ner, tokenize, lemma, pos",
        "ner",

    )
    def test_spacy_variation_pipeline(self, value):
        spacy = Pipeline[DataPack]()
        spacy.set_reader(StringReader())

        config = {
            "processors": value,
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }
        spacy.add(SpacyProcessor(), config=config)
        spacy.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack: DataPack = spacy.process(document)
        tokens: List[Token] = list(pack.get_entries(Token))  # type: ignore

        raw_results = self.nlp(document)
        sentences = raw_results.sents

        if "tokenize" in value:
            exp_pos = []
            exp_lemma = []
            for s in sentences:
                for w in s:
                    exp_lemma.append(w.lemma_)
                    exp_pos.append(w.tag_)

            tokens_text = [x.text for x in tokens]
            self.assertEqual(tokens_text, document.replace('.', ' .').split())

            pos = [x.pos for x in tokens]
            lemma = [x.lemma for x in tokens]

            # Check token texts
            for token, text in zip(tokens, tokens_text):
                start, end = token.span.begin, token.span.end
                self.assertEqual(document[start:end], text)

            if "pos" in value:
                self.assertListEqual(pos, exp_pos)
            else:
                none_pos = [None] * len(pos)
                self.assertListEqual(pos, none_pos)

            if "lemma" in value:
                self.assertListEqual(lemma, exp_lemma)
            else:
                none_lemma = [None] * len(lemma)
                self.assertListEqual(lemma, none_lemma)
        else:
            self.assertListEqual(tokens, [])

        if "ner" in value:
            pack_ents: List[EntityMention] = list(
                pack.get_entries(EntityMention))
            entities_text = [x.text for x in pack_ents]
            entities_type = [x.ner_type for x in pack_ents]

            raw_ents = raw_results.ents
            exp_ent_text = [
                document[ent.start_char: ent.end_char] for ent in raw_ents
            ]
            exp_ent_types = [ent.label_ for ent in raw_ents]

            self.assertEqual(entities_text, exp_ent_text)
            self.assertEqual(entities_type, exp_ent_types)

    def test_neg_spacy_processor(self):
        spacy = Pipeline[DataPack]()
        spacy.set_reader(StringReader())

        config = {
            "processors": 'ner',
            "lang": "xx_ent_wiki_sm",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }
        spacy.add(SpacyProcessor(), config=config)
        spacy.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        with self.assertRaises(ProcessExecutionException):
            _ = spacy.process(document)


if __name__ == "__main__":
    unittest.main()
