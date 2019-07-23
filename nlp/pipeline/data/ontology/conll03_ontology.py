# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.base_ontology import *


class Token(Token):  # type: ignore
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.chunk_tag = None
        self.ner_tag = None
