import forte.data.ontology.base_ontology as ontology
from typing import List

class Word(ontology.Token):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.dependency_relation = None
        self.feats = None
        self.governor = None
        self.index = None
        self.lemma = None
        self.pos = None
        self.text = None
        self.upos = None
        self.xpos = None
        self.pos_tag = None

class Token(ontology.Token):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.index = None
        self.text = None
        self.words = None

class Sentence(ontology.Sentence):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.dependencies= None
        self.tokens = None  # type: List[Token]
        self.words = None  # type: List[Word]
