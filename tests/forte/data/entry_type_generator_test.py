import unittest
from unittest.mock import patch
from forte.data.entry_type_generator import EntryTypeGenerator
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
    EntityMention,
    PredicateArgument,
    PredicateLink,
    PredicateMention,
    CoreferenceGroup,
)

class EntryTypeGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    # @patch("forte.data.entry_type_generator._get_type_attributes")
    def test_get_type_attributes(self):
        # test that _get_type_attributes() is called only once
        type_attributes = EntryTypeGenerator.get_type_attributes()
        self.assertLessEqual(len(type_attributes['Annotation']['Token']), 9)
        self.assertLessEqual(len(type_attributes['Annotation']['Sentence']), 5)
        # type_attributes = self.entry_type_generator.get_type_attributes()
        # self.assertLessEqual(mock_get_type_attributes.call_count, 1)
        # EntryTypeGenerator.get_type_attributes()
        # self.assertLessEqual(mock_get_type_attributes.call_count, 1)

if __name__ == "__main__":
    unittest.main()