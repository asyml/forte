# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from forte.data.ontology.base_ontology import *
from forte.data.data_pack import DataPack


class Token(Token):  # type: ignore
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.sense = None


class Sentence(Sentence):  # type: ignore
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.speaker = None
        self.part_id = None


class PredicateMention(PredicateMention):  # type: ignore
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pred_type = None
        self.pred_lemma = None
        self.framenet_id = None
