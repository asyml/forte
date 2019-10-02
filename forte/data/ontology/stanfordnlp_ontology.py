import forte.data.ontology.base_ontology as ontology
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Link

__all__ = [
    "Token",
    "Document",
    "Sentence"
]


class Token(ontology.Token):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.lemma: str
        self.upos: str
        self.xpos: str
        self.pos_tag: str


class Sentence(ontology.Sentence):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Document(ontology.Document):
    # TODO: code generation method will help reduce such problems.
    # pylint: disable=useless-super-delegation
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Dependency(Link):  # pylint: disable=too-many-ancestors
    ParentType = Token
    ChildType = Token

    def __init__(self, pack: DataPack, parent: Token, child: Token):
        super().__init__(pack, parent, child)
        self.rel_type: str
