# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from forte.data.data_pack import DataPack
from forte.data.ontology.base_ontology import (
    Document, Sentence, Token, EntityMention)

__all__ = [
    "Document",
    "Sentence",
    "Token",
    "EntityMention",
]


class Token(Token):  # type: ignore
    """
    A subclass of :class:`~forte.data.ontology.base_ontology.Token`.
    Includes token fields that are specific to CoNLL03 dataset.

    Args:
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.chunk_tag: str
        self.ner_tag: str
