import unittest
from unittest.mock import patch
from forte.data.entry_type_generator import EntryTypeGenerator
from forte.data.ontology.top import Annotation, Generics
from typing import Optional, Dict
from forte.data.data_pack import DataPack
from dataclasses import dataclass


@dataclass
class TokenTest(Annotation):
    """
    A span based annotation :class:`Tokentest`, used to represent a token or a word.
    Attributes:
        pos (Optional[str]):
        ud_xpos (Optional[str]):
        lemma (Optional[str]):
        chunk (Optional[str]):
        ner (Optional[str]):
        sense (Optional[str]):
        is_root (Optional[bool]):
        ud_features (Dict[str, str]):
        ud_misc (Dict[str, str]):
    """

    pos: Optional[str]
    ud_xpos: Optional[str]
    lemma: Optional[str]
    chunk: Optional[str]
    ner: Optional[str]
    sense: Optional[str]
    is_root: Optional[bool]
    ud_features: Dict[str, str]
    ud_misc: Dict[str, str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.ud_xpos: Optional[str] = None
        self.lemma: Optional[str] = None
        self.chunk: Optional[str] = None
        self.ner: Optional[str] = None
        self.sense: Optional[str] = None
        self.is_root: Optional[bool] = None
        self.ud_features: Dict[str, str] = dict()
        self.ud_misc: Dict[str, str] = dict()


@dataclass
class TitleTest(Annotation):
    """
    A span based annotation `Title`, normally used to represent a title.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class MetricTest(Generics):
    """
    A base metric entity, all metric entities should inherit from it.
    Attributes:
        metric_name (Optional[str]):
    """

    metric_name: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.metric_name: Optional[str] = None


@dataclass
class SingleMetricTest(MetricTest):
    """
    A single metric entity, used to present a metric of one float (e.g. accuracy).
    Attributes:
        value (Optional[float]):
    """

    value: Optional[float]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: Optional[float] = None


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
            "entry_type_generator_test.TokenTest": [
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
            "entry_type_generator_test.TitleTest": [],
            "entry_type_generator_test.SingleMetricTest": [
                "metric_name",
                "value",
            ],
        }
        for entry_name in entry_name_attributes_dict.keys():
            attribute_result = EntryTypeGenerator.get_entry_attributes_by_class(
                entry_name
            )
            self.assertEqual(
                attribute_result, entry_name_attributes_dict[entry_name]
            )


if __name__ == "__main__":
    unittest.main()
