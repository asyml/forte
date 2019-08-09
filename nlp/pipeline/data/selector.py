"""
This defines some selector interface used as glue to combine
DataPack/multiPack processors and Pipeline.
"""
from typing import List, Generic
from nlp.pipeline.data import PackType
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.multi_pack import MultiPack


class Selector(Generic[PackType]):
    def __init__(self, **kwargs):
        pass

    def select(self, data_pack: PackType):
        raise NotImplementedError


class DummySelector(Selector):
    """
    Do nothing, return the data pack itself, which can be either DataPack
    or MultiPack
    """

    def select(self, data_pack: PackType) -> PackType:
        return data_pack


class SinglePackSelector(Selector[MultiPack]):
    """
    Select a DataPack from a MultiPack with specified name
    """

    def __init__(self, select_name: str):
        super().__init__()
        self.select_name = select_name

    def select(self, data_pack: MultiPack) -> DataPack:
        if self.select_name not in data_pack.packs:
            raise ValueError(f"pack name {self.select_name}"
                             f"not in the MultiPack")
        assert self.select_name is not None
        return data_pack.packs[self.select_name]


class MultiPackSelector(Selector[MultiPack]):
    """
    Select multiple packs from the input MultiPack with specified names,
    the select function returns a MultiPack
    """

    def __init__(self, select_names: List[str]):
        super().__init__()
        self.select_names = select_names

    def select(self, data_pack: MultiPack) -> MultiPack:
        ret_pack: MultiPack = MultiPack()
        for name in self.select_names:
            if name not in data_pack.packs:
                raise ValueError(f"pack name {name}"
                                 f"not in the MultiPack")
            ret_pack.update_pack(**{name: data_pack.packs[name]})
        return ret_pack
