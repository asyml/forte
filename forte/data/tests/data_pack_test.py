"""
Unit tests for data pack related operations.
"""
import logging
import os
import unittest

import forte
from ft.onto.base_ontology import (
    Token, Sentence, Document, EntityMention, PredicateArgument, PredicateLink,
    PredicateMention)
from forte.data.readers import OntonotesReader

logging.basicConfig(level=logging.DEBUG)


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        self.reader = OntonotesReader()
        data_path = os.path.join(os.path.dirname(
            os.path.dirname(forte.__file__)),
            "forte/data/tests/data_samples/ontonotes/00/abc_0059.gold_conll"
        )
        self.data_pack = list(self.reader.parse_pack(data_path))[0]

    def test_get_data(self):
        requests = {
            Sentence: ["speaker"],
            Token: ["pos", "sense"],
            EntityMention: [],
            PredicateMention: [],
            PredicateArgument: {
                "fields": [],
                "unit": "Token"
            },
            PredicateLink: {
                "component": self.reader.component_name,
                "fields": ["parent", "child", "arg_type"]
            }
        }

        # case 1: get sentence context from the beginning
        instances = list(self.data_pack.get_data(Sentence))
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[1]["offset"],
                         len(instances[0]["context"]) + 1)

        # case 2: get sentence context from the second instance
        instances = list(self.data_pack.get_data(Sentence, skip_k=1))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 165)

        # case 3: get document context
        instances = list(self.data_pack.get_data(Document, skip_k=0))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 0)

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data(Sentence, skip_k=10))
        self.assertEqual(len(instances), 0)

        # case 5: get entries
        instances = list(self.data_pack.get_data(Sentence,
                                                 request=requests,
                                                 skip_k=1))
        self.assertEqual(len(instances[0].keys()), 9)
        self.assertEqual(len(instances[0]["PredicateLink"]), 4)
        self.assertEqual(len(instances[0]["Token"]), 5)
        self.assertEqual(len(instances[0]["EntityMention"]), 3)

        # case 6: test delete entry
        num_sent = len([self.data_pack.get(Sentence)])
        first_sent = [self.data_pack.get(Sentence)][0]
        self.data_pack.delete_entry(first_sent)
        self.assertEqual(len([self.data_pack.get_data(Sentence)]), num_sent - 1)


if __name__ == '__main__':
    unittest.main()
