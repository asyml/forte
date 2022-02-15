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
import warnings

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
    def __init__(self):
        self.configs: Config = Config({}, {})

    def select(self, pack: InputPackType) -> Iterator[OutputPackType]:
        raise NotImplementedError

    def initialize(
        self, configs: Optional[Union[Config, Dict[str, Any]]] = None
    ):
        self.configs = self.make_configs(configs)


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

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        reverse = self.configs.reverse_selection

        for name, pack in m_pack.iter_packs():
            if reverse:
                if not self.will_select(name, pack, m_pack):
                    yield pack
            else:
                if self.will_select(name, pack, m_pack):
                    yield pack

    def will_select(
        self, pack_name: str, pack: DataPack, multi_pack: MultiPack
    ) -> bool:
        """
        Implement this method to return a boolean value whether the
        pack will be selected.

        Args:
            pack_name: The name of the pack to be selected.
            pack: The pack that needed to be determined whether it will be
              selected.
            multi_pack: The original multi pack.

        Returns: A boolean value to indicate whether `pack` will be returned.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"reverse_selection": False}


class NameMatchSelector(SinglePackSelector):
    r"""
    Select a :class:`DataPack` from a :class:`MultiPack` with specified
    name. This implementation takes special care for backward compatibility.

    Deprecated:
        .. code-block:: python

            selector = NameMatchSelector(select_name="foo")
            selector = NameMatchSelector("foo")

    Now:
        .. code-block:: python

            selector = NameMatchSelector()
                selector.initialize(
                    configs={
                        "select_name": "foo"
                    }
            )

    WARNING: Passing parameters through __init__ is deprecated, and does not
    work well with pipeline serialization.
    """

    def __init__(self, select_name: Optional[str] = None):
        super().__init__()
        self.select_name = select_name
        if select_name is not None:
            warnings.warn(
                (
                    "Passing parameters through __init__ is deprecated,"
                    " and does not work well with pipeline serialization."
                )
            )

    def will_select(
        self, pack_name: str, pack: DataPack, multi_pack: MultiPack
    ):
        return pack_name == self.select_name

    def initialize(
        self, configs: Optional[Union[Config, Dict[str, Any]]] = None
    ):
        super().initialize(configs)
        try:
            configs_ = configs.todict()  # type:ignore
        except AttributeError:
            configs_ = {} if configs is None else configs

        if self.select_name is not None:
            configs_["select_name"] = self.select_name
        super().initialize(configs_)

        if self.configs["select_name"] is None:
            raise ValueError("select_name shouldn't be None.")
        self.select_name = self.configs["select_name"]

    @classmethod
    def default_configs(cls):
        return {"select_name": None}


class RegexNameMatchSelector(SinglePackSelector):
    r"""Select a :class:`DataPack` from a :class:`MultiPack` using a regex.

    This implementation takes special care for backward compatibility.

    Deprecated:
        .. code-block:: python

            selector = RegexNameMatchSelector(select_name="^.*\\d$")
            selector = RegexNameMatchSelector("^.*\\d$")

    Now:
        .. code-block:: python

            selector = RegexNameMatchSelector()
            selector.initialize(
                configs={
                    "select_name": "^.*\\d$"
                }
            )

    WARNING:
        Passing parameters through __init__ is deprecated, and does not
        work well with pipeline serialization.

    """

    def __init__(self, select_name: Optional[str] = None):
        super().__init__()
        self.select_name = select_name
        if select_name is not None:
            warnings.warn(
                (
                    "Passing parameters through __init__ is deprecated,"
                    " and does not work well with pipeline serialization."
                )
            )

    def will_select(
        self, pack_name: str, pack: DataPack, multi_pack: MultiPack
    ) -> bool:
        return re.match(self.select_name, pack_name) is not None  # type:ignore

    def initialize(
        self, configs: Optional[Union[Config, Dict[str, Any]]] = None
    ):
        super().initialize(configs)

        try:
            configs_ = configs.todict()  # type:ignore
        except AttributeError:
            configs_ = {} if configs is None else configs

        if self.select_name is not None:
            configs_["select_name"] = self.select_name

        super().initialize(configs_)

        if self.configs["select_name"] is None:
            raise ValueError("select_name shouldn't be None.")
        self.select_name = self.configs["select_name"]

    @classmethod
    def default_configs(cls):
        return {"select_name": None}


class FirstPackSelector(SinglePackSelector):
    r"""Select the first entry from :class:`MultiPack` and yield it."""

    def will_select(
        self, pack_name: str, pack: DataPack, multi_pack: MultiPack
    ) -> bool:
        return multi_pack.pack_names[0] == pack_name


class AllPackSelector(SinglePackSelector):
    r"""Select all the packs from :class:`MultiPack` and yield them."""

    def will_select(
        self, pack_name: str, pack: DataPack, multi_pack: MultiPack
    ) -> bool:
        return True
