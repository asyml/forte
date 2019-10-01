"""
Standard Universal Dependency Format -
https://universaldependencies.org/docs/format.html
"""
from typing import Dict, List, Optional

from forte.data.ontology.base_ontology import (Token, Sentence, Document,
                                               Dependency)
from forte.data.data_pack import DataPack

__all__ = [
    "Sentence",
    "Document",
    "DependencyToken",
    "UniversalDependency"
]


class DependencyToken(Token):
    """
        Token type for dependency parsing containing additional information
        than base_ontology.Token
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        # self._tid defined as a part of Entry
        # self.text defined as a part of Annotation
        self.is_root: bool = False
        self.universal_pos_tag: Optional[str] = None
        self.features: Dict[str, List[str]] = {}
        self.lemma: Optional[str] = None
        self.language_pos_tag: Optional[str] = None
        self.misc: Optional[Dict[str, List[str]]] = None


class UniversalDependency(Dependency):
    """
    Dependency Link type for dependency parsing containing
    additional "dependency type" (primary or enhanced) information
    """
    def __init__(self,
                 pack: DataPack,
                 parent: Optional[DependencyToken] = None,
                 child: Optional[DependencyToken] = None):
        super().__init__(pack, parent, child)
        self.dep_type: Optional[str] = None
