"""
Unit tests for Pipeline.
"""
import unittest

from texar.torch import HParams

from forte.data.readers import OntonotesReader
from forte.pipeline import Pipeline
from forte.processors.base.tests.dummy_batch_processor import *
from ft.onto.base_ontology import Token, Sentence, RelationLink


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        # Define and config the Pipeline
        self.nlp = Pipeline()
        self.nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = HParams({"batcher": {"batch_size": 5}},
                         dummy.default_hparams())
        self.nlp.add_processor(dummy, config=config)
        self.nlp.initialize()

        self.dataset_path = \
            "forte/tests/data_samples/ontonotes_sample_dataset/00"

    def test_process_next(self):
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                sent_text = sentence.text

                # first method to get entry in a sentence
                for link in pack.get_entries(RelationLink, sentence):
                    parent = link.get_parent()
                    child = link.get_child()
                    print(f"{parent.text} is {link.rel_type} {child.text}")
                    pass  # some operation on link

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))


if __name__ == '__main__':
    unittest.main()
