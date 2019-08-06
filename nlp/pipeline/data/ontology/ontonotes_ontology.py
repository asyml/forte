# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.base_ontology import *


class Token(Token):  # type: ignore
    """
    A subclass of :class:`~nlp.pipeline.data.ontology.base_ontology.Token`.
    Includes token fields that are specific to Ontonotes dataset.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.sense = None


class Sentence(Sentence):  # type: ignore
    """
    A subclass of :class:`~nlp.pipeline.data.ontology.base_ontology.Sentence`.
    Includes sentence fields that are specific to Ontonotes dataset.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.speaker = None
        self.part_id = None


class PredicateMention(PredicateMention):  # type: ignore
    """
    A subclass of
    :class:`~nlp.pipeline.data.ontology.base_ontology.PredicateMention`.
    Includes predicate mention fields that are specific to Ontonotes dataset.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.pred_type = None
        self.pred_lemma = None
        self.framenet_id = None
