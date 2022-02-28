import unittest
from unittest.mock import patch
from forte.data.entry_type_generator import EntryTypeGenerator


class EntryTypeGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @patch("forte.data.entry_type_generator._get_type_attributes")
    def test_get_type_attributes(self, mock_get_type_attributes):
        # test that _get_type_attributes() is called only once
        EntryTypeGenerator.get_type_attributes()
        EntryTypeGenerator.get_type_attributes()
        self.assertLessEqual(mock_get_type_attributes.call_count, 1)
        EntryTypeGenerator.get_type_attributes()
        self.assertLessEqual(mock_get_type_attributes.call_count, 1)

    def test_get_entry_attribute_by_class(self):
        entry_name_attributes_dict = {
            "ft.onto.base_ontology.Sentence": [
                "speaker",
                "part_id",
                "sentiment",
                "classification",
                "classifications",
            ],
            "ft.onto.base_ontology.Token": [
                "pos",
                "ud_xpos",
                "lemma",
                "chunk",
                "ner",
                "sense",
                "is_root",
                "ud_features",
                "ud_misc",
            ],
            "ft.onto.base_ontology.Title": [],
            "ft.onto.metric.SingleMetric": ["value"]
        }
        for entry_name in entry_name_attributes_dict.keys():
            attribute_result = EntryTypeGenerator.get_entry_attribute_by_class(
                entry_name
            )
            self.assertEqual(
                attribute_result, entry_name_attributes_dict[entry_name]
            )


if __name__ == "__main__":
    unittest.main()
