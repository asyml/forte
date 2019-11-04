"""
Unit tests for dummy processor.
"""
import unittest
from ddt import ddt, data

from texar.torch import HParams

from forte.data.readers import OntonotesReader, StringReader
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.processors.base.tests.dummy_batch_processor import \
    DummyRelationExtractor, DummmyFixedSizeBatchProcessor
from ft.onto.base_ontology import RelationLink, Sentence


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = Pipeline()
        self.nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = HParams({"batcher": {"batch_size": 5}},
                         dummy.default_hparams())
        self.nlp.add_processor(dummy, config=config)
        self.nlp.initialize()

        self.data_path = \
            "forte/processors/base/tests/data_samples/ontonotes/00/"

    def test_processor(self):
        pack = self.nlp.process(self.data_path)

        relations = list(pack.get_entries(RelationLink))

        assert (len(relations) > 0)

        for relation in relations:
            self.assertEqual(relation.get_field("rel_type"), "dummy_relation")


@ddt
class DummyFixedSizeBatchProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = Pipeline()
        self.nlp.set_reader(StringReader())
        self.dummy = DummmyFixedSizeBatchProcessor()

    @data(1, 2, 3)
    def test_processor(self, batch_size):
        config = HParams({"batcher": {"batch_size": batch_size}},
                         self.dummy.default_hparams())
        self.nlp.add_processor(NLTKSentenceSegmenter())
        self.nlp.add_processor(self.dummy, config=config)
        self.nlp.initialize()
        sentences = ["This tool is called Forte. The goal of this project to "
                     "help you build NLP pipelines. NLP has never been made "
                     "this easy before."]
        pack = self.nlp.process(sentences)
        sent_len = len(list(pack.get(Sentence)))
        self.assertEqual(
            self.dummy.counter, (sent_len // batch_size +
                                 (sent_len % batch_size > 0)))


if __name__ == '__main__':
    unittest.main()
