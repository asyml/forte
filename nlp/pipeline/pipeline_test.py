"""
Unit tests for Pipeline.
"""
import unittest
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.data.readers import OntonotesReader
from nlp.pipeline.processors.impl.dummy_processor import *
from nlp.pipeline.data.ontology import relation_ontology
from nlp.pipeline.data.ontology.relation_ontology import *


class PipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        # Define and config the Pipeline
        self.dataset_path = "examples/ontonotes_sample_dataset/00"

        self.nlp = Pipeline(ontology=relation_ontology)

        self.nlp.set_reader(OntonotesReader())
        self.processor = DummyRelationExtractor()
        self.nlp.add_processor(self.processor)

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
