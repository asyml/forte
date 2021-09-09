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
from typing import Generic, Iterator, TypeVar, Optional, Union, Dict, Any

import re

from forte.common.configuration import Config
from forte.common.configurable import Configurable
from forte.data.base_pack import BasePack
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack

InputPackType = TypeVar("InputPackType", bound=BasePack)
OutputPackType = TypeVar("OutputPackType", bound=BasePack)

__all__ = [
    "Selector",
    "DummySelector",
    "SinglePackSelector",
    "NameMatchSelector",
    "RegexNameMatchSelector",
    "FirstPackSelector",
    "AllPackSelector",
]


class Selector(Generic[InputPackType, OutputPackType], Configurable):
    def __init__(self, configs: Optional[Union[Config, Dict[str, Any]]] = None):
        self.configs = self.make_configs(configs)

    def select(self, pack: InputPackType) -> Iterator[OutputPackType]:
        raise NotImplementedError


class DummySelector(Selector[InputPackType, InputPackType]):
    r"""Do nothing, return the data pack itself, which can be either
    :class:`DataPack` or :class:`MultiPack`.
    """

    def select(self, pack: InputPackType) -> Iterator[InputPackType]:
        yield pack


class SinglePackSelector(Selector[MultiPack, DataPack]):
    """
    This is the base class that select a DataPack from MultiPack.
    """

    def select(self, pack: MultiPack) -> Iterator[DataPack]:
        raise NotImplementedError


class NameMatchSelector(SinglePackSelector):
    r"""Select a :class:`DataPack` from a :class:`MultiPack` with specified
    name.

    This implementation takes special care for backward compatability:
    Deprecated:
        selector = NameMatchSelector(select_name="foo")
        selector = NameMatchSelector("foo")
    Now:
        selector = NameMatchSelector(
            configs={
                "select_name": "foo"
            }
        )
    """

    def __init__(self, *args, **kwargs):
        assert (len(args) == 0) ^ (len(kwargs) == 0)
        if args:
            configs = {"select_name": args[0]}
        else:
            assert ("configs" in kwargs) or ("select_name" in kwargs)
            if "select_name" in kwargs:
                configs = {"select_name": kwargs["select_name"]}
            else:
                configs = kwargs["configs"]
        super().__init__(configs=configs)
        self.select_name = self.configs["select_name"]
        assert self.select_name is not None

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        matches = 0
        for name, pack in m_pack.iter_packs():
            if name == self.select_name:
                matches += 1
                yield pack

        if matches == 0:
            raise ValueError(
                f"Pack name {self.select_name}" f" not in the MultiPack"
            )

    @classmethod
    def default_configs(cls):
        return {"select_name": None}


class RegexNameMatchSelector(SinglePackSelector):
    r"""Select a :class:`DataPack` from a :class:`MultiPack` using a regex.

    This implementation takes special care for backward compatability:
    Deprecated:
        selector = RegexNameMatchSelector(select_name="^.*\\d$")
        selector = RegexNameMatchSelector("^.*\\d$")
    Now:
        selector = RegexNameMatchSelector(
            configs={
                "select_name": "^.*\\d$"
            }
        )
    """

    def __init__(self, *args, **kwargs):
        assert (len(args) == 0) ^ (len(kwargs) == 0)
        if args:
            configs = {"select_name": args[0]}
        else:
            assert ("configs" in kwargs) or ("select_name" in kwargs)
            if "select_name" in kwargs:
                configs = {"select_name": kwargs["select_name"]}
            else:
                configs = kwargs["configs"]
        super().__init__(configs=configs)
        self.select_name = self.configs["select_name"]
        assert self.select_name is not None

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError("Multi-pack is empty")
        else:
            for name, pack in m_pack.iter_packs():
                if re.match(self.select_name, name):
                    yield pack

    @classmethod
    def default_configs(cls):
        return {"select_name": None}


class FirstPackSelector(SinglePackSelector):
    r"""Select the first entry from :class:`MultiPack` and yield it."""

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError("Multi-pack has no data packs.")

        else:
            yield m_pack.packs[0]


class AllPackSelector(SinglePackSelector):
    r"""Select all the packs from :class:`MultiPack` and yield them."""

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        if len(m_pack.packs) == 0:
            raise ValueError("Multi-pack has no data packs.")

        else:
            yield from m_pack.packs
