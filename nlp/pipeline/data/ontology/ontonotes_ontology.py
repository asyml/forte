# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.base_ontology import *


class Token(Token):
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.sense = None


class Sentence(Sentence):
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.speaker = None
        self.part_id = None


class PredicateMention(PredicateMention):
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.pred_type = None
        self.pred_lemma = None
        self.framenet_id = None
