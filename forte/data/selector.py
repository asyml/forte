"""
This defines some selector interface used as glue to combine
DataPack/multiPack processors and Pipeline.
"""
from typing import Generic, Iterator, TypeVar

from forte.data import BasePack
from forte.data import DataPack
from forte.data import MultiPack

InputPackType = TypeVar('InputPackType', bound=BasePack)
OutputPackType = TypeVar('OutputPackType', bound=BasePack)


class Selector(Generic[InputPackType, OutputPackType]):
    def __init__(self, **kwargs):
        pass

    def select(self, pack: InputPackType) -> Iterator[OutputPackType]:
        raise NotImplementedError


class DummySelector(Selector[InputPackType, InputPackType]):
    """
    Do nothing, return the data pack itself, which can be either DataPack
    or MultiPack
    """

    def select(self, pack: InputPackType) -> Iterator[InputPackType]:
        yield pack


class SinglePackSelector(Selector[MultiPack, DataPack]):
    def select(self, pack: MultiPack) -> Iterator[DataPack]:
        raise NotImplementedError


class NameMatchSelector(SinglePackSelector):
    """
    Select a DataPack from a MultiPack with specified name
    """

    def __init__(self, select_name: str):
        super().__init__()
        assert self.select_name is not None
        self.select_name: str = select_name

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        matches = 0
        for name, pack in m_pack.iter_packs():
            if name == self.select_name:
                matches += 1
                yield pack

        if matches == 0:
            raise ValueError(f"pack name {self.select_name}"
                             f"not in the MultiPack")
