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
profiling test/demo for attribute interface (of data pack):
"""

import os
import unittest

__all__ = [
    "performance_test",
]

from typing import Optional, Dict, Set, List, Any, Iterator

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.caster import MultiPackBoxer

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

    def _process_old(self, input_pack: DataPack):
        token_entries = list(
            input_pack.get(entry_type=Token, components=self.token_component)
        )
        token_texts = [token.text for token in token_entries]
        taggings = pos_tag(token_texts)
        for token, tag in zip(token_entries, taggings):
            token.pos = tag[1]

    def _process(self, input_pack: DataPack):
        token_entries = list(
            input_pack.get_attributes_of_type(
                "ft.onto.base_ontology.Token",
                ["begin", "end", "pos"],
                components=self.token_component)
        )

        token_texts = []
        for token, attrs_from_ds in token_entries:
            token_text = input_pack.get_span_text(attrs_from_ds['begin'], attrs_from_ds['end'])
            token_texts.append(token_text)

        taggings = pos_tag(token_texts)
        for token_entry, tag in zip(token_entries, taggings):
            token = token_entry[0]
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

    def testPOSTaggingNER(self, input_path: str = ''):  # input_output_pair ,
        """
        Verify the intermediate representation of pipeline.
        """
        input_path = (
            "/Users/jamesxiao/Downloads/Semantic-Role-Labeling-master/conll-formatted-ontonotes-5.0/"
            "data/conll-2012-test/data/english/annotations/bc/phoenix/00/"  # bn/"
        )
        if len(input_path) == 0:
            self.nlp.set_reader(StringReader())
            input_param = (
                "Forte is a data-centric ML framework. Muad Dib learned \
                rapidly because his first training was in how to learn. "
                "And the first lesson of all was the basic trust that he \
                could learn. "
                "It's shocking to find how many people do not believe they \
                can learn, and how many more believe learning to be difficult."
            )
        else:
            self.nlp.set_reader(OntonotesReader())
            input_param = input_path
        self.nlp.add(NLTKSentenceSegmenter())
        self.nlp.add(NLTKWordTokenizer())
        self.nlp.add(NLTKPOSTagger())
        self.nlp.add(NLTKNER())

        for pack in self.nlp.initialize().process_dataset(
                input_param
        ):  # initialize().run(input_path):   #:  rs:  #
            for sentence in pack.get("ft.onto.base_ontology.Sentence"):
                print("The sentence is: ", sentence.text)
                print("The POS tags of the tokens are:")
                for token in pack.get(Token, sentence):
                    print(f" {token.text}[{token.pos}]", end=" ")
                print()

    def testSimpleSerialization(self, input_path: str = ''):
        """
        Verify the intermediate representation of pipeline.
        """
        input_path = (
            "/Users/jamesxiao/Downloads/Semantic-Role-Labeling-master/conll-formatted-ontonotes-5.0/"
            "data/conll-2012-test/data/english/annotations/bc/phoenix/00/"
        )
        output_path = "./test_simple_pack_output/"

        if len(input_path) == 0:
            self.nlp.set_reader(StringReader())
            input_param = (
                "Forte is a data-centric ML framework. Muad Dib learned \
                rapidly because his first training was in how to learn. "
                "And the first lesson of all was the basic trust that he \
                could learn. "
                "It's shocking to find how many people do not believe they \
                can learn, and how many more believe learning to be difficult."
            )
        else:
            self.nlp.set_reader(OntonotesReader())
            input_param = input_path

        self.nlp.add(
            PackNameJsonPackWriter(),
            {
                "output_dir": output_path,
                "indent": 2,
                "overwrite": True,
            },
        )

        self.nlp.run(input_param)

        coref_pl = Pipeline()
        coref_pl.set_reader(DirPackReader())
        coref_pl.add(MultiPackBoxer())

        # initialize().process_dataset   input_param
        coref_pl.run(
            output_path
        )

        # for pack in self.nlp.initialize().process_dataset(
        #         output_path #input_param
        # ):  # initialize().run(input_path):   #:  rs:  #
        #     for sentence in pack.get("ft.onto.base_ontology.Sentence"):
        #         print("The sentence is: ", sentence.text)
        #         print("The POS tags of the tokens are:")
        #         for token in pack.get(Token, sentence):
        #             print(f" {token.text}[{token.pos}]", end=" ")
        #         print()


def define_skip_condition(flag: str, explanation: str):
    return unittest.skipUnless(
        os.environ.get(flag, 0) or os.environ.get("TEST_ALL", 0),
        explanation + f" Set `{flag}=1` or `TEST_ALL=1` to run.",
    )

performance_test = define_skip_condition(
    "TEST_PERFORMANCE", "Test/demo the performance of new attribute interface."
)
