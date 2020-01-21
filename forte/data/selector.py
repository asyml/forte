# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This defines some selector interface used as glue to combine
DataPack/multiPack processors and Pipeline.
"""
import re
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
    r"""Do nothing, return the data pack itself, which can be either DataPack or
    MultiPack
    """

    def select(self, pack: InputPackType) -> Iterator[InputPackType]:
        yield pack


class SinglePackSelector(Selector[MultiPack, DataPack]):
    def select(self, pack: MultiPack) -> Iterator[DataPack]:
        raise NotImplementedError


class NameMatchSelector(SinglePackSelector):
    r"""Select a DataPack from a MultiPack with specified name"""

    def __init__(self, select_name: str):
        super().__init__()
        assert select_name is not None
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


class RegexNameMatchSelector(SinglePackSelector):
    r"""
    Select a DataPack from a MultiPack using a regex.
    """
    def __init__(self, select_name: str):
        super().__init__()
        assert select_name is not None
        self.select_name: str = select_name

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError("Multi-pack is empty")

        else:
            for name, pack in m_pack.iter_packs():
                if re.match(self.select_name, name):
                    yield pack


class FirstPackSelector(SinglePackSelector):
    r"""Select the first entry from the multi-pack and yield it.
    """

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError(f"Multi-pack has no data packs.")

        else:
            yield m_pack.packs[0]


class AllPackSelector(SinglePackSelector):
    r"""Select all the packs from the multi-pack and yield them
    """

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError(f"Multi-pack has no data packs.")

        else:
            yield from m_pack.packs
