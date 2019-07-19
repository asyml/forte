# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.base_ontology import *


class Token(Token):  # type: ignore
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.sense = None


class Sentence(Sentence):  # type: ignore
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.speaker = None
        self.part_id = None


class PredicateMention(PredicateMention):  # type: ignore
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.pred_type = None
        self.pred_lemma = None
        self.framenet_id = None
