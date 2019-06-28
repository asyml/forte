"""
Unit tests for Pipeline.
"""
import unittest
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.data.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.processors.dummy_processor import *


class PipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        # Define and config the Pipeline
        reader = OntonotesReader()
        dataset_path = "/Users/wei.wei/Documents/nlp-pipeline" \
                       "/conll-formatted-ontonotes-5.0/data/" \
                       "test/data/english/annotations/bn/abc/"
        self.data_packs_iter = reader.dataset_iterator(dataset_path)

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
        for pack in self.nlp.process_next():
            # get sentence from pack
            for sentence in pack.get_entries(OntonotesOntology.Sentence):
                sent_text = sentence.text

                # first method to get entry in a sentence
                for link in pack.get_entries(RelationOntology.RelationLink,
                                             sentence):
                    pass # some operation on link

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(OntonotesOntology.Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))


if __name__ == '__main__':
    unittest.main()
