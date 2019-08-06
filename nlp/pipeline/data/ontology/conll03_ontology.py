# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.base_ontology import *


class Token(Token):  # type: ignore
    """
    A subclass of :class:`~nlp.pipeline.data.ontology.base_ontology.Token`.
    Includes token fields that are specific to CoNLL03 dataset.

    Args:
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.chunk_tag = None
        self.ner_tag = None
