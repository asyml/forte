# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from forte.data.data_pack import DataPack
from forte.data.ontology.base_ontology import *


class Token(Token):  # type: ignore
    """
    A subclass of :class:`~forte.data.ontology.base_ontology.Token`.
    Includes token fields that are specific to Ontonotes dataset.

    Args:
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.sense = None


class Sentence(Sentence):  # type: ignore
    """
    A subclass of :class:`~forte.data.ontology.base_ontology.Sentence`.
    Includes sentence fields that are specific to Ontonotes dataset.

     Args:
        begin (int): The offset of the first character in the sentence.
        end (int): The offset of the last character in the sentence + 1.

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.speaker = None
        self.part_id = None


class PredicateMention(PredicateMention):  # type: ignore
    """
    A subclass of
    :class:`~forte.data.ontology.base_ontology.PredicateMention`.
    Includes predicate mention fields that are specific to Ontonotes dataset.

    Args:
        begin (int): The offset of the first character in the predicate mention.
        end (int): The offset of the last character in the predicate mention
            + 1.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pred_type = None
        self.pred_lemma = None
        self.framenet_id = None
