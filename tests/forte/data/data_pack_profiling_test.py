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
Utils for unit tests.
"""

import os
import re
import unittest
import nltk

__all__ = [
    "performance_test",
]

from typing import Any, Callable

from typing import Optional, Dict, Set, List, Any, Iterator

from forte.common.configuration import Config
from forte.common.resources import Resources

from forte.data.readers import OntonotesReader, DirPackReader, StringReader
from forte.processors.writers import PackNameJsonPackWriter

from ft.onto.base_ontology import EntityMention, Token, Sentence, Phrase
from nltk import (  # type: ignore
    pos_tag,
    ne_chunk,
    PunktSentenceTokenizer,
    download,
)

from nltk.tokenize import sent_tokenize, word_tokenize

from forte.processors.base import PackProcessor
from forte.data.data_pack import DataPack

from forte import Pipeline
from nltk.tokenize.treebank import TreebankWordTokenizer

# from fortex.spacy import SpacyProcessor


class SentenceAndTokenProcessor(PackProcessor):
    def __init__(self):
        super().__init__()

    def initialize(self, resources, configs):
        super().initialize(resources, configs)

    def process_tokens(self, sentences, input_pack: DataPack):
        """Basic tokenization and post tagging of the sentence.
        Args:
            processors: List of processor names.
            sentences: Generator object which yields sentences in document.
            input_pack: input pack which needs to be modified.
        Returns: A mapping from SpaCy token index to Forte Token.
        """
        tokens: [Token] = []

        last_sentence_word_idx = 0
        for s_idx, sentence in sentences:
            Sentence(input_pack, s_idx, s_idx + len(sentence))

            for word in sentence:
                begin_pos_word = word.idx
                end_pos_word = begin_pos_word + len(word.text)
                token = Token(input_pack, begin_pos_word, end_pos_word)
                tokens.append(token)

        return tokens

    def _process(self, input_pack: DataPack):
        doc = input_pack.text

        sentences = sent_tokenize(doc)

        # tokens = process_tokens(sentences, input_pack)   # sentences, input_pack
        tokens: [Token] = []

        last_sentence_word_idx = 0
        s_idx = 0
        for sentence in sentences:
            e_idx = s_idx + len(sentence)
            Sentence(input_pack, s_idx, e_idx)

            last_sentence_word_idx = s_idx
            for word in word_tokenize(sentence):
                begin_pos_word = last_sentence_word_idx
                end_pos_word = begin_pos_word + len(word)
                token = Token(input_pack, begin_pos_word, end_pos_word)
                last_sentence_word_idx = end_pos_word + 1
                tokens.append(token)

            s_idx = e_idx + 1

        return tokens

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`. The processor produce
        different types with different settings of `processors` in config.
        Args:
            record_meta: the field in the data pack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Sentence"] = set()
        record_meta["ft.onto.base_ontology.Token"] = set()


class ExampleNLTKPOSTagger(PackProcessor):
    r"""A wrapper of NLTK pos tagger."""

    def initialize(self, resources, configs):
        super().initialize(resources, configs)
        # download the NLTK average perceptron tagger
        nltk.download("averaged_perceptron_tagger")

    def _process(self, input_pack: DataPack):
        # get a list of token data entries from `input_pack`
        # using `DataPack.get()`` method

        token_texts = [token.text for token in input_pack.get(Token)]

        # use nltk pos tagging module to tag token texts
        taggings = nltk.pos_tag(token_texts)

        # assign nltk taggings to token attributes
        for token, tag in zip(input_pack.get(Token), taggings):
            token.pos = tag[1]

            # token.pos = word.tag_

            # token.lemma = word.lemma_

            # Store the spacy token index to forte token mapping.
            # indexed_tokens[word.i] = token

        # return indexed_tokens

    def record(record_meta: Dict[str, Set[str]]):
        record_meta["ft.onto.base_ontology.Token"].add("pos")
        record_meta["ft.onto.base_ontology.Token"].add("lemma")

    def process_tokens(
        processors, sentences, input_pack: DataPack
    ) -> Dict[int, Token]:
        """Basic tokenization and post tagging of the sentence.
        Args:
            processors: List of processor names.
            sentences: Generator object which yields sentences in document.
            input_pack: input pack which needs to be modified.
        Returns: A mapping from SpaCy token index to Forte Token.
        """
        indexed_tokens: Dict[int, Token] = {}

        for sentence in sentences:
            Sentence(input_pack, sentence.start_char, sentence.end_char)

            if "tokenize" in processors:
                # Iterating through spaCy token objects
                for word in sentence:
                    begin_pos_word = word.idx
                    end_pos_word = begin_pos_word + len(word.text)
                    token = Token(input_pack, begin_pos_word, end_pos_word)

                    if "pos" in processors:
                        token.pos = word.tag_

                    if "lemma" in processors:
                        token.lemma = word.lemma_

                    # Store the spacy token index to forte token mapping.
                    indexed_tokens[word.i] = token
        return indexed_tokens


class NLTKNER(PackProcessor):
    r"""A wrapper of NLTK NER."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        download("maxent_ne_chunker")
        download("words")

    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(
                input_pack.get(
                    entry_type=Token,
                    range_annotation=sentence,
                    components=self.token_component,
                )
            )
            tokens = [(token.text, token.pos) for token in token_entries]
            ne_tree = ne_chunk(tokens)

            index = 0
            for chunk in ne_tree:
                if hasattr(chunk, "label"):
                    # For example:
                    # chunk: Tree('GPE', [('New', 'NNP'), ('York', 'NNP')])
                    begin_pos = token_entries[index].span.begin
                    end_pos = token_entries[index + len(chunk) - 1].span.end
                    entity = EntityMention(input_pack, begin_pos, end_pos)
                    entity.ner_type = chunk.label()
                    index += len(chunk)
                else:
                    # For example:
                    # chunk: ('This', 'DT')
                    index += 1

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKNER` which is
        `ft.onto.base_ontology.EntityMention` with attribute `phrase_type`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.EntityMention"] = {"ner_type"}

    def expected_types_and_attributes(self):
        r"""Method to add expected type ft.onto.base_ontology.Token` with
        attribute `pos` and `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {
            "ft.onto.base_ontology.Sentence": set(),
            "ft.onto.base_ontology.Token": {"pos"},
        }


class NLTKWordTokenizer(PackProcessor):
    r"""A wrapper of NLTK word tokenizer."""

    def __init__(self):
        super().__init__()
        self.tokenizer = TreebankWordTokenizer()

    def _process(self, input_pack: DataPack):
        for begin, end in self.tokenizer.span_tokenize(input_pack.text):
            Token(input_pack, begin, end)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKWordTokenizer`, which is
        `ft.onto.base_ontology.Token`,
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Token"] = set()


class NLTKSentenceSegmenter(PackProcessor):
    r"""A wrapper of NLTK sentence tokenizer."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        download("punkt")

    def __init__(self):
        super().__init__()
        self.sent_splitter = PunktSentenceTokenizer()

    def _process(self, input_pack: DataPack):
        for begin, end in self.sent_splitter.span_tokenize(input_pack.text):
            Sentence(input_pack, begin, end)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKSentenceSegmenter`, which
        is `ft.onto.base_ontology.Sentence`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Sentence"] = set()


class NLTKPOSTagger(PackProcessor):
    r"""A wrapper of NLTK pos tagger."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        download("averaged_perceptron_tagger")

    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        token_entries = list(
            input_pack.get(entry_type=Token, components=self.token_component)
        )
        token_texts = [token.text for token in token_entries]
        taggings = pos_tag(token_texts)
        for token, tag in zip(token_entries, taggings):
            token.pos = tag[1]

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKPOSTagger`, which adds
        attribute `pos` to `ft.onto.base_ontology.Token`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Token"].add("pos")

    def expected_types_and_attributes(self):
        r"""Method to add expected type `ft.onto.base_ontology.Token` for input
        which would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {"ft.onto.base_ontology.Token": set()}


class NLP_Pipeline_Performance_Test(unittest.TestCase):
    """
    Test performance of POS, NER.
    """

    def setUp(self) -> None:
        self.nlp = Pipeline[DataPack]()
        # self.nlp.set_reader(StringReader())

    def test_POS_tagging(self):  # input_output_pair , , input_path : str
        """
        Verify the intermediate representation of pipeline.
        """
        pack_output = "pack_out"
        input_path = (
            "/Users/jamesxiao/Downloads/Semantic-Role-Labeling-master/conll-formatted-ontonotes-5.0/"
            "data/conll-2012-test/data/english/annotations/bc/phoenix/00/"
        )  # msnbc_0007.gold_conll

        self.nlp.set_reader(OntonotesReader())
        # self.nlp.set_reader(StringReader())
        self.nlp.add(NLTKSentenceSegmenter())  # SentenceAndTokenProcessor
        self.nlp.add(NLTKWordTokenizer())
        self.nlp.add(NLTKPOSTagger())  #  #ExampleNLTKPOSTagger()

        # self.nlp.add(SentenceAndTokenProcessor())  #, {"processors": ["sentence", "tokenize"]}
        # self.nlp.add(ExampleNLTKPOSTagger())

        # self.nlp.add(
        #     PackNameJsonPackWriter(),
        #     {
        #         "output_dir": pack_output,
        #         "indent": 2,
        #         "overwrite": True,
        #     },
        # )

        input_string = (
            "Forte is a data-centric ML framework. Muad Dib learned rapidly because his first training was in how to learn. "
            "And the first lesson of all was the basic trust that he could learn. "
            " It's shocking to find how many people do not believe they can learn, and how many more believe learning to be difficult. "
        )

        # self.nlp.initialize()
        # rs = self.nlp.run(input_path)
        for pack in self.nlp.initialize().process_dataset(
            input_path
        ):  # initialize().run(input_path):   #:  rs:  #
            for sentence in pack.get("ft.onto.base_ontology.Sentence"):
                print("The sentence is: ", sentence.text)
                print("The POS tags of the tokens are:")
                for token in pack.get(Token, sentence):
                    print(f" {token.text}[{token.pos}]", end=" ")
                print()


def define_skip_condition(flag: str, explanation: str):
    return unittest.skipUnless(
        os.environ.get(flag, 0) or os.environ.get("TEST_ALL", 0),
        explanation + f" Set `{flag}=1` or `TEST_ALL=1` to run.",
    )


performance_test = define_skip_condition(
    "TEST_PERFORMANCE", "Test the performance of Forte modules."
)
