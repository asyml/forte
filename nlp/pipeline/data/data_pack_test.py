"""
Unit tests for ner_data pack related operations.
"""
import os
import unittest
import nlp
from nlp.pipeline.data.readers import OntonotesOntology, OntonotesReader
from nlp.pipeline.utils import *


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        self.reader = OntonotesReader()
        data_path = os.path.join(os.path.dirname(
            os.path.dirname(nlp.__file__)), "examples/abc_0059.gold_conll")
        self.data_pack = self.reader.read(data_path)

    def test_coverage_index(self):
        # case 1: automatically built one-type to all-type index
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 1)
        self.assertIn("Sentence-to-Entry",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Sentence-to-Entry"]
        self.assertEqual(len(cov_index["Sentence.0"]), 27 + 22 + 10)  # 10 links
        self.assertEqual(len(cov_index["Sentence.1"]), 12 + 15 + 6)  # 6 links

        # case 2: build one-type to one-type index
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 1)
        self.data_pack.index.build_coverage_index(
            self.data_pack.annotations,
            outer_type=OntonotesOntology.Sentence,
            inner_type=OntonotesOntology.Token
        )
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 2)
        self.assertIn("Sentence-to-Token",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Sentence-to-Token"]
        self.assertEqual(len(cov_index["Sentence.0"]), 27)
        self.assertEqual(len(cov_index["Sentence.1"]), 12)

        # case 3: build all-type to one-type index
        self.data_pack.index.build_coverage_index(
            self.data_pack.annotations,
            inner_type=OntonotesOntology.Token
        )
        self.assertEqual(len(self.data_pack.index.coverage_index.keys()), 3)
        self.assertIn("Annotation-to-Token",
                      self.data_pack.index.coverage_index.keys())

        cov_index = self.data_pack.index.coverage_index["Annotation-to-Token"]
        self.assertEqual(len(cov_index.keys()),
                         27 + 22 + 12 + 15)  # annotation num
        self.assertEqual(len(cov_index["Sentence.1"]), 12)

    def test_get_data(self):
        antype = {
            "Sentence": ["speaker"],
            "Token": ["pos_tag", "sense"],
            "EntityMention": [],
            "PredicateMention": [],
            "PredicateArgument": {
                "fields": [],
                "unit": "Token"
            },
        }
        linktype = {
            "PredicateLink": {
                "component": self.reader.component_name,
                "fields": ["parent","child", "arg_type"]
            }
        }

        # case 1: get sentence context from the beginning
        instances = list(self.data_pack.get_data("sentence"))
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[1]["offset"],
                         len(instances[0]["context"]) + 1)

        # case 2: get sentence context from the second instance
        instances = list(self.data_pack.get_data("sentence", offset=1))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 165)

        # case 3: get document context
        instances = list(self.data_pack.get_data("Document", offset=0))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 0)

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data("sentence", offset=10))
        self.assertEqual(len(instances), 0)

        # case 5: get entries
        instances = list(self.data_pack.get_data("sentence",
                                                 annotation_types=antype,
                                                 link_types=linktype,
                                                 offset=1))
        self.assertEqual(len(instances[0].keys()), 9)
        self.assertEqual(len(instances[0]["PredicateLink"]), 4)
        self.assertEqual(len(instances[0]["Token"]), 5)
        self.assertEqual(len(instances[0]["EntityMention"]), 3)

        # case 5: get batch
        batch_size = 2
        instances = list(self.data_pack.get_data_batch(batch_size=batch_size,
                                                       context_type="sentence",
                                                       annotation_types=antype,
                                                       link_types=linktype))
        self.assertEqual(len(instances[0][0].keys()), 9)
        self.assertEqual(len(instances[0][0]["Token"]), 5)
        self.assertEqual(len(instances[0][0]["EntityMention"]), 3)
        self.assertEqual(len(instances[0][0]["Token"]["text"]), batch_size)


if __name__ == '__main__':
    unittest.main()
