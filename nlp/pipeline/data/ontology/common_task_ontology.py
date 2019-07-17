from nlp.pipeline.data.ontology.base_ontology import BaseOntology

__all__ = [
    "CoNLL03Ontology",
    "OntonotesOntology"
]


class CoNLL03Ontology(BaseOntology):
    """
        The ontology for CoNLL03 dataset
    """
    class Token(BaseOntology.Token):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.chunk_tag = None
            self.pos_tag = None
            self.ner_tag = None


class OntonotesOntology(BaseOntology):
    """
    The ontology for Ontonotes dataset
    """
    class Token(BaseOntology.Token):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.sense = None
            self.pos_tag = None

    class Sentence(BaseOntology.Sentence):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.speaker = None
            self.part_id = None

    class PredicateMention(BaseOntology.PredicateMention):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.pred_type = None
            self.pred_lemma = None
            self.framenet_id = None
