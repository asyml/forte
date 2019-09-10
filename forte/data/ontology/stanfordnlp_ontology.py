import forte.data.ontology.base_ontology as ontology
from forte.data.ontology.top import Link

__all__ = [
    "Token",
    "Document",
    "Sentence"
]


class Token(ontology.Token):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.dependency_relation = None
        self.governor = None
        self.lemma = None
        self.upos = None
        self.xpos = None
        self.pos_tag = None


class Sentence(ontology.Sentence):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.tokens = None


class Document(ontology.Document):
    # TODO: code generation method will help reduce such problems.
    # pylint: disable=useless-super-delegation
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)


class Dependency(Link):
    def __init__(self, parent: Token, child: Token):
        super().__init__(parent, child)
        self.rel_type: str
