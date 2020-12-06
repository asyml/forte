"""This module tests LowerCaser processor."""
import unittest
from typing import List

from ddt import ddt, data, unpack

from allennlp.predictors import Predictor

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.allennlp_processors import AllenNLPProcessor, MODEL2URL
from forte.processors.spacy_processors import SpacyProcessor
from ft.onto.base_ontology import Sentence, Token, Dependency
from forte.common import ProcessorConfigError, ProcessExecutionException


@ddt
class TestAllenNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.allens = {
            # TODO: Current download model is wrong on Allennlp.
            # 'universal': Predictor.from_path(MODEL2URL['universal']),
            'stanford': Predictor.from_path(MODEL2URL['stanford'])
        }

        self.results = {}
        for k in self.allens:
            self.results[k] = {}

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP pipelines.",
            "NLP has never been made this easy before.",
            "Forte is named Forte because it is designed for text."
        ]
        self.document = ' '.join(sentences)

        for k in self.allens:
            self.results[k]['tokens'] = []
            self.results[k]['pos'] = []
            self.results[k]['dep_types'] = []
            self.results[k]['dep_heads'] = []

        for dep_type in self.allens.keys():
            for sent in sentences:
                results = self.allens[dep_type].predict(  # type: ignore
                    sentence=sent)

                self.results[dep_type]['tokens'].append(results['words'])
                self.results[dep_type]['pos'].append(results['pos'])
                self.results[dep_type]['dep_types'].append(
                    results['predicted_dependencies'])
                self.results[dep_type]['dep_heads'].append(
                    results['predicted_heads'])

    @data(
        "tokenize",
        "tokenize,pos",
        "tokenize,pos,depparse",
        "tokenize,depparse",
        "",
        "pos",  # nothing will be output by processor
        "depparse",  # nothing will be output by processor
    )
    def test_allennlp_processor_with_different_processors(self, processors):
        nlp = self._create_pipeline({
            'processors': processors
        })
        pack = nlp.process(self.document)

        if processors == "":
            processors = AllenNLPProcessor.default_configs()['processors']
        tag_format = AllenNLPProcessor.default_configs()['tag_formalism']

        self._check_results(pack, processors, tag_format)

    @data(
        "stanford",
        # "universal",  # TODO: Current download model is wrong on Allennlp.
        "random_dependencies",
    )
    def test_allennlp_processor_with_different_tag_formats(self, format):
        if format == "random_dependencies":
            with self.assertRaises(ProcessorConfigError):
                self._create_pipeline({'tag_formalism': format})
        else:
            nlp = self._create_pipeline({'tag_formalism': format})
            pack = nlp.process(self.document)

            processors = AllenNLPProcessor.default_configs()['processors']

            self._check_results(pack, processors, format)

    @data(
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )
    @unpack
    def test_allennlp_processor_with_existing_entries(self, overwrite_entries,
                                                      allow_parallel_entries):
        config = {
            'overwrite_entries': overwrite_entries,
            'allow_parallel_entries': allow_parallel_entries
        }
        nlp = self._create_pipeline(config)

        # Adding extra processor to have existing tokens and dependencies
        nlp.add(component=AllenNLPProcessor(), config=config)
        nlp.initialize()

        if not overwrite_entries and not allow_parallel_entries:
            # Processor should raise config error when both the flags are False
            # and existing entries are found
            with self.assertRaises(ProcessExecutionException):
                nlp.process(self.document)
        else:
            pack = nlp.process(self.document)

            processors = AllenNLPProcessor.default_configs()['processors']
            tag_format = AllenNLPProcessor.default_configs()['tag_formalism']

            if not overwrite_entries:
                if allow_parallel_entries:
                    # Should raise AssertionError due to duplicate tokens
                    with self.assertRaises(AssertionError):
                        self._check_results(pack, processors, tag_format)
            else:
                self._check_results(pack, processors, tag_format)

    def _check_results(self, pack, processors, tag_format):
        # checking the whole datapack text
        self.assertEqual(pack.text, self.document)

        if "tokenize" in processors:
            for i, sentence in enumerate(pack.get(Sentence)):
                # checking the tokens and pos
                tokens = self._test_tokenizer(pack, sentence, i,
                                              processors, tag_format)

                if "depparse" in processors:
                    deps: List[Dependency] = list(
                        pack.get(Dependency, sentence))

                    indexed_deps = {}
                    for d in deps:
                        indexed_deps[d.get_child().tid] = d

                    sorted_deps = []
                    for t in tokens:
                        sorted_deps.append(indexed_deps[t.tid])

                    # checking the dependencies
                    self._test_dependencies(i, tokens, sorted_deps, tag_format)

    @staticmethod
    def _create_pipeline(config):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(StringReader())

        # Using SpacyProcessor to segment the sentences
        nlp.add(component=SpacyProcessor(), config={
            'processors': '',
            'lang': "en_core_web_sm",  # Language code to build the Pipeline
            'use_gpu': False
        })

        nlp.add(component=AllenNLPProcessor(), config=config)
        nlp.initialize()
        return nlp

    def _test_tokenizer(self, pack, sentence, sent_idx,
                        processors, tag_format):
        tokens = []
        for j, token in enumerate(
                pack.get(entry_type=Token, range_annotation=sentence)):
            self.assertEqual(
                token.text, self.results[tag_format]['tokens'][sent_idx][j])
            self._test_pos(sent_idx, token, j, processors, tag_format)
            tokens.append(token)
        return tokens

    def _test_pos(self, sent_idx, token, token_idx,
                  processors, tag_format):
        exp_pos = self.results[tag_format]['pos'][sent_idx][token_idx] \
            if "pos" in processors else None
        self.assertEqual(token.pos, exp_pos)

    def _test_dependencies(self, sent_idx, tokens, deps, tag_format):
        print(deps)
        for j, dep in enumerate(deps):
            self.assertEqual(
                dep.get_parent(),
                tokens[self.results[tag_format]['dep_heads'][sent_idx][j] - 1])
            self.assertEqual(
                dep.rel_type,
                self.results[tag_format]['dep_types'][sent_idx][j])
