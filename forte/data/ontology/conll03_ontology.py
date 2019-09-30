# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from forte.data.data_pack import DataPack
from forte.data.ontology.base_ontology import Document, Sentence, Token

__all__ = [
    "Document",
    "Sentence",
    "Token",
]


class Token(Token):  # type: ignore
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.chunk_tag: str
        self.ner_tag: str
