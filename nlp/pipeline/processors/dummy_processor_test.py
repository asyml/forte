"""
Unit tests for data iterator related operations.
"""
import unittest
from nlp.pipeline.io.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.processors.dummy_processor import DummyRelationExtractor
from nlp.pipeline.pipeline import Pipeline


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        reader = OntonotesReader()
        dataset_path = "/Users/wei.wei/Documents/nlp-pipeline" \
                       "/conll-formatted-ontonotes-5.0/data/" \
                       "test/data/english/annotations/bn/abc/"
        self.data_packs_iter = reader.dataset_iterator(dataset_path)

        self.annotype = {
            "Token": [],
            "EntityMention": ["ner_type"]
        }

        self.processor = DummyRelationExtractor()
        self.processor.annotation_types = self.annotype
        self.processor.context_type = "sentence"
        self.processor.batch_size = 4

    def test_processor(self):

        # case 1: process data
        data_pack = next(self.data_packs_iter)
        instances = list(data_pack.get_data(self.processor.context_type,
                                            self.processor.annotation_types))
        out_dict = self.processor.process(instances[0])
        self.assertEqual(len(out_dict["RelationshipLink"].keys()), 3)
        for k in out_dict["RelationshipLink"].keys():
            self.assertEqual(len(out_dict["RelationshipLink"][k]), 1)

    def test_pipeline(self):
        kwargs = {
            "dataset": {
                "dataset_dir": "/Users/wei.wei/Documents/nlp-pipeline"
                               "/conll-formatted-ontonotes-5.0/data/" 
                               "test/data/english/annotations/bn/abc/",
                "dataset_format": "Ontonotes"
            }
        }

        nlp = Pipeline(**kwargs)
        nlp.processors.append(self.processor)
        nlp._processors_beginning.append((0,0))
        out_dict = nlp.process_next()

        self.assertEqual(len(out_dict["RelationshipLink"].keys()), 3)
        for k in out_dict["RelationshipLink"].keys():
            self.assertEqual(len(out_dict["RelationshipLink"][k]),
                             self.processor.batch_size)


if __name__ == '__main__':
    unittest.main()
