"""
Unit tests for Pipeline.
"""
import unittest
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors.dummy_processor import *


class PipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        # Define and config the Pipeline
        dataset_path = "/Users/wei.wei/Documents/nlp-pipeline" \
                       "/conll-formatted-ontonotes-5.0/data/" \
                       "test/data/english/annotations/bn/abc/"

        kwargs = {
            "dataset": {
                "dataset_dir": dataset_path,
                "dataset_format": "Ontonotes"
            }
        }
        self.nlp = Pipeline(**kwargs)

        self.processor = DummyRelationExtractor()
        self.nlp.processors.append(self.processor)

    def test_process_next(self):

        # get processed pack from dataset
        for pack in self.nlp.process_dataset(hard_batch=False):
            # get sentence from pack
            for sentence in pack.get_entries(RelationOntology.Sentence):
                sent_text = sentence.text

                # first method to get entry in a sentence
                for link in pack.get_entries(RelationOntology.RelationLink,
                                             sentence):
                    parent = link.get_parent()
                    child = link.get_child()
                    print(f"{parent.text} is {link.rel_type} {child.text}")
                    pass  # some operation on link

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(RelationOntology.Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))


if __name__ == '__main__':
    unittest.main()
