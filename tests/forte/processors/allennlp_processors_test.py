"""This module tests LowerCaser processor."""
import unittest
from ddt import ddt, data, unpack

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.allennlp_processors import AllenNLPProcessor
from forte.processors.spacy_processors import SpacyProcessor
from ft.onto.base_ontology import Sentence, Token, Dependency
from forte.common import ProcessorConfigError


@ddt
class TestAllenNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.document = "This tool is called Forte. The goal of this project " \
                        "to help you build NLP pipelines. NLP has never been " \
                        "made this easy before."
        self.tokens = [["This", "tool", "is", "called", "Forte", "."],
                       ["The", "goal", "of", "this", "project", "to", "help",
                        "you", "build", "NLP", "pipelines", "."],
                       ["NLP", "has", "never", "been", "made", "this", "easy",
                        "before", "."]]
        self.pos = {
            'stanford_dependencies': [
                ['DT', 'NN', 'VBZ', 'VBN', 'NNP', '.'],
                ['DT', 'NN', 'IN', 'DT', 'NN', 'TO', 'VB', 'PRP', 'VB', 'NNP',
                 'NNS', '.'],
                ['NNP', 'VBZ', 'RB', 'VBN', 'VBN', 'DT', 'JJ', 'RB', '.'],
            ],
            'universal_dependencies': [
                ['DET', 'NOUN', 'AUX', 'VERB', 'PROPN', 'PUNCT'],
                ['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PART', 'VERB', 'PRON',
                 'VERB', 'PROPN', 'NOUN', 'PUNCT'],
                ['PROPN', 'AUX', 'ADV', 'AUX', 'VERB', 'DET', 'ADJ', 'ADV',
                 'PUNCT'],
            ],
        }
        self.deps = {
            'stanford_dependencies': [
                ['det', 'nsubjpass', 'auxpass', 'root', 'xcomp', 'punct'],
                ['det', 'root', 'prep', 'det', 'pobj', 'aux', 'infmod', 'nsubj',
                 'ccomp', 'nn', 'dobj', 'punct'],
                ['nsubjpass', 'aux', 'neg', 'auxpass', 'root', 'det', 'xcomp',
                 'advmod', 'punct'],
            ],
            'universal_dependencies': [
                ['det', 'nsubj:pass', 'aux:pass', 'root', 'xcomp', 'punct'],
                ['det', 'root', 'case', 'det', 'nmod', 'mark', 'acl', 'obj',
                 'xcomp', 'compound', 'obj', 'punct'],
                ['nsubj:pass', 'aux', 'advmod', 'aux:pass', 'root', 'det',
                 'obj', 'advmod', 'punct'],
            ]

        }
        self.dep_heads = {
            'stanford_dependencies': [
                [2, 4, 4, 0, 4, 4],
                [2, 0, 2, 5, 3, 7, 5, 9, 7, 11, 9, 2],
                [5, 5, 5, 5, 0, 7, 5, 5, 5],
            ],
            'universal_dependencies': [
                [2, 4, 4, 0, 4, 4],
                [2, 0, 5, 5, 2, 7, 2, 7, 7, 11, 9, 2],
                [5, 5, 5, 5, 0, 7, 5, 5, 5],
            ]
        }

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
        output_format = AllenNLPProcessor.default_configs()['output_format']

        self._check_results(pack, processors, output_format)

    @data(
        "stanford_dependencies",
        "universal_dependencies",
        "random_dependencies",
    )
    def test_allennlp_processor_with_different_output_formats(self, format):
        if format == "random_dependencies":
            with self.assertRaises(ProcessorConfigError):
                self._create_pipeline({'output_format': format})
        else:
            nlp = self._create_pipeline({'output_format': format})
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
        nlp.add_processor(processor=AllenNLPProcessor(), config=config)
        nlp.initialize()

        if not overwrite_entries and not allow_parallel_entries:
            # Processor should raise config error when both the flags are False
            # and existing entries are found
            with self.assertRaises(ProcessorConfigError):
                nlp.process(self.document)
        else:
            pack = nlp.process(self.document)

            processors = AllenNLPProcessor.default_configs()['processors']
            output_format = AllenNLPProcessor.default_configs()['output_format']

            if not overwrite_entries:
                if allow_parallel_entries:
                    # Should raise AssertionError due to duplicate tokens
                    with self.assertRaises(AssertionError):
                        self._check_results(pack, processors, output_format)
            else:
                self._check_results(pack, processors, output_format)

    @data(
        "This tool is called Forte tool.",
        "NLP NLP NLP NLP.",
        "AllenNLP does NLP.",
    )
    def test_allennlp_processor_with_repeating_words(self, sentence):
        processors = "tokenize"
        nlp = self._create_pipeline({
            'processors': processors
        })
        self.document = sentence
        self.tokens = [sentence.replace('.', ' .').split()]
        pack = nlp.process(self.document)

        output_format = AllenNLPProcessor.default_configs()['output_format']

        self._check_results(pack, processors, output_format)

    def _check_results(self, pack, processors, output_format):
        # checking the whole datapack text
        self.assertEqual(pack.text, self.document)

        if "tokenize" in processors:
            deps = [dep for dep in pack.get(Dependency)]
            offset = 0
            for i, sentence in enumerate(pack.get(Sentence)):
                # checking the tokens and pos
                tokens = self._test_tokenizer(pack, sentence, i,
                                              processors, output_format)

                if "depparse" in processors:
                    # checking the dependencies
                    self._test_dependencies(i, tokens, deps, offset,
                                            output_format)
                    offset += len(self.tokens[i])

    @staticmethod
    def _create_pipeline(config):
        nlp = Pipeline()
        nlp.set_reader(StringReader())

        # Using SpacyProcessor to segment the sentences
        nlp.add_processor(processor=SpacyProcessor(), config={
            'processors': '',
            'lang': "en_core_web_sm",  # Language code to build the Pipeline
            'use_gpu': False
        })

        nlp.add_processor(processor=AllenNLPProcessor(), config=config)
        nlp.initialize()
        return nlp

    def _test_tokenizer(self, pack, sentence, sent_idx,
                        processors, output_format):
        tokens = []
        for j, token in enumerate(
                pack.get(entry_type=Token, range_annotation=sentence)):
            self.assertEqual(token.text, self.tokens[sent_idx][j])
            self._test_pos(sent_idx, token, j, processors, output_format)
            tokens.append(token)
        return tokens

    def _test_pos(self, sent_idx, token, token_idx,
                  processors, output_format):
        exp_pos = self.pos[output_format][sent_idx][token_idx] \
            if "pos" in processors else None
        self.assertEqual(token.pos, exp_pos)

    def _test_dependencies(self, sent_idx, tokens, deps, offset, output_format):
        for j, dep in enumerate(deps[offset:offset +
                                            len(self.tokens[sent_idx])]):
            self.assertEqual(dep.get_parent(),
                tokens[self.dep_heads[output_format][sent_idx][j] - 1])
            self.assertEqual(dep.rel_type,
                             self.deps[output_format][sent_idx][j])
