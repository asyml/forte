"""
Standard Universal Dependency Format -
https://universaldependencies.org/docs/format.html
"""
from typing import Dict, List, Optional

from forte.data.ontology.base_ontology import (Token, Sentence, Document,
                                               Dependency)

__all__ = [
    "Sentence",
    "Document",
    "DependencyToken",
    "DependencyLink"
]


class DependencyToken(Token):
    """
        Token type for dependency parsing containing additional information
        than base_ontology.Token
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        # self._tid defined as a part of Entry
        # self.text defined as a part of Annotation
        self.root = False
        self.universal_pos_tag = None
        self.features = None
        # optional fields
        self.lemma: Optional[str] = None
        self.language_pos_tag: Optional[str] = None
        self.misc: Optional[Dict[str, List[str]]] = None


class DependencyLink(Dependency):
    """
    Dependency Link type for dependency parsing containing
    additional "type" (regular or enhanced) information than
    base_ontology.Dependency
    """
    def __init__(self,
                 parent: Optional[DependencyToken] = None,
                 child: Optional[DependencyToken] = None):
        super().__init__(parent, child)
        self.type = None
