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
        entry_name = "ft.onto.base_ontology.Sentence"
        results = EntryTypeGenerator.get_entry_attribute_by_class(entry_name)
        print(results)


if __name__ == "__main__":
    unittest.main()