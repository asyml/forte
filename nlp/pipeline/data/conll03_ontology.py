"""
This class defines the ontology for CoNLL03 dataset
"""

from nlp.pipeline.data.base_ontology import BaseOntology

__all__ = [
    "CoNLL03Ontology",
]


class CoNLL03Ontology(BaseOntology):
    class Token(BaseOntology.Token):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.chunk_tag = None
            self.pos_tag = None
            self.ner_tag = None
