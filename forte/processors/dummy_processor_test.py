"""
Unit tests for dummy processor.
"""
import os
import unittest

from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.pipeline import Pipeline
from forte.processors.dummy_batch_processor import DummyRelationExtractor
from forte.data.ontology import relation_ontology


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = Pipeline()
        self.nlp.set_ontology(relation_ontology)
        self.reader = OntonotesReader()

        self.data_path = "examples/data_samples/ontonotes/00/"

        self.nlp.set_reader(OntonotesReader())
        self.nlp.add_processor(DummyRelationExtractor())
        self.nlp.initialize()

    def test_processor(self):
        pack = self.nlp.process(self.data_path)

        relations = list(pack.get_entries(relation_ontology.RelationLink))

        assert (len(relations) > 0)

        for relation in relations:
            assert (relation.get_field("rel_type") == "dummy_relation")


if __name__ == '__main__':
    unittest.main()
