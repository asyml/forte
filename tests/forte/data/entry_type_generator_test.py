import unittest
from unittest.mock import patch
from forte.data.entry_type_generator import EntryTypeGenerator
# from ft.onto.base_ontology import Token
# from ft.onto.wikipedia import WikiPage
# from ft.onto.metric import Metric
# from ftx.onto.race_qa import Passage
# from ftx.onto.ag_news import Description
# from ftx.medical import UMLSConceptLink
from ontology.test_outputs.ft.onto.test_top_attribute import Item

class EntryTypeGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.type_attributes = EntryTypeGenerator().get_type_attributes()

    @patch("forte.data.entry_type_generator._get_type_attributes")
    def test_get_type_attributes(self, mock_get_type_attributes):
        # test that _get_type_attributes() is called only once
        EntryTypeGenerator.get_type_attributes()
        EntryTypeGenerator.get_type_attributes()
        self.assertLessEqual(mock_get_type_attributes.call_count, 1)
        EntryTypeGenerator.get_type_attributes()
        self.assertLessEqual(mock_get_type_attributes.call_count, 1)

    def test_get_type_attributes_all(self):
        # test the results
        # print(self.type_attributes["Annotation"])
        self.assertEqual(len(self.type_attributes["Annotation"]), 26)
        self.assertEqual(len(self.type_attributes["Link"]), 5)
        self.assertEqual(len(self.type_attributes["Group"]), 1)
        self.assertEqual(len(self.type_attributes["Annotation"]["Token"]), 9)
        self.assertEqual(len(self.type_attributes["Annotation"]["Sentence"]), 5)
        self.assertEqual(len(self.type_attributes["Link"]["Dependency"]), 2)


if __name__ == "__main__":
    unittest.main()
