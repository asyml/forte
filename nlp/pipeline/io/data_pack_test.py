"""
Unit tests for data pack related operations.
"""
import os
import unittest
import nlp
from nlp.pipeline.io.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.io.ontonotes_ontology import *


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        reader = OntonotesReader()
        data_path = os.path.join(os.path.dirname(os.path.dirname(nlp.__file__)),
                                 "examples/abc_0059.gold_conll")
        self.data_pack = reader.read(data_path)

    def test_coverage_index(self):

        # case 1: build one-type to one-type index
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 0)
        self.data_pack.index.build_coverage_index(self.data_pack.annotations,
                                                  outer_type=OntonotesOntology.Sentence,
                                                  inner_type=OntonotesOntology.Token)
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 1)
        self.assertIn("Sentence-to-Token",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Sentence-to-Token"]
        self.assertEqual(len(cov_index["Sentence.0"]), 27)
        self.assertEqual(len(cov_index["Sentence.1"]), 12)

        # case 2: build all-type to one-type index
        self.data_pack.index.build_coverage_index(self.data_pack.annotations,
                                                  inner_type=OntonotesOntology.Token)
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 2)
        self.assertIn("Annotation-to-Token",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Annotation-to-Token"]
        self.assertEqual(len(cov_index.keys()), 27+22+12+15)  # annotation num
        self.assertEqual(len(cov_index["Sentence.1"]), 12)

        # case 3: build one-type to all-type index
        self.data_pack.index.build_coverage_index(self.data_pack.annotations,
                                                  links=self.data_pack.links,
                                                  groups=self.data_pack.groups,
                                                  outer_type=OntonotesOntology.Sentence)
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 3)
        self.assertIn("Sentence-to-Entry",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Sentence-to-Entry"]
        self.assertEqual(len(cov_index["Sentence.0"]), 27+22+10)  # 10 links
        self.assertEqual(len(cov_index["Sentence.1"]), 12+15+6)  # 6 links


if __name__ == '__main__':
    unittest.main()
