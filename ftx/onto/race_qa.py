# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/forte/ontology_specs/race_qa.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology race_multi_choice_qa_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional

__all__ = [
    "Passage",
]


@dataclass
class Passage(Annotation):
    """
    Attributes:
        passage_id (Optional[str]):
    """

    passage_id: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.passage_id: Optional[str] = None
