""" This class defines the ontology for Conll03-NER dataset
"""

from nlp.pipeline.io.base_ontology import BaseOntology


class Conll03NER(BaseOntology):
    class Token(BaseOntology.Token):
        def __init__(self, component: str, begin: int, end: int, tid: str = None):
            super().__init__(component, begin, end, tid)
            self.chunk_tag = None
            self.pos_tag = None
            self.ner_tag = None
