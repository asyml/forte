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
import yaml

from forte.common.configuration import Config
from forte.common import SelectorConfigError
from forte.data.base_pack import BasePack
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.utils import get_full_module_name

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


class Selector(Generic[InputPackType, OutputPackType]):
    def __init__(self, 
                 configs: Optional[Union[Config, Dict[str, Any]]] = None):
        self.configs = self.make_configs(configs)

    def select(self, pack: InputPackType) -> Iterator[OutputPackType]:
        raise NotImplementedError

    @classmethod
    def make_configs(
        cls, configs: Optional[Union[Config, Dict[str, Any]]]
    ) -> Config:
        """
        Create the component configuration for this class, by merging the
        provided config with the ``default_configs()``.

        The following config conventions are expected:
          - The top level key can be a special `config_path`.
          - `config_path` should be point to a file system path, which will
             be a YAML file containing configurations.
          - Other key values in the configs will be considered as parameters.

        Args:
            configs: The input config to be merged with the default config.

        Returns:
            The merged configuration.
        """
        merged_configs: Dict = {}

        if configs is not None:
            if isinstance(configs, Config):
                configs = configs.todict()

            if "config_path" in configs and not configs["config_path"] is None:
                filebased_configs = yaml.safe_load(
                    open(configs.pop("config_path"))
                )
            else:
                filebased_configs = {}

            merged_configs.update(filebased_configs)

            merged_configs.update(configs)

        try:
            final_configs = Config(merged_configs, cls.default_configs())
        except ValueError as e:
            raise SelectorConfigError(
                f"Configuration error for the selector "
                f"{get_full_module_name(cls)}."
            ) from e

        return final_configs

    @classmethod
    def default_configs(cls):
        r"""Returns a `dict` of configurations of the component with default
        values. Used to replace the missing values of input `configs`
        during selector construction.
        """
        return {}


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

    def __init__(self, 
                 configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super.__init__(configs)

    def select(self, pack: MultiPack) -> Iterator[DataPack]:
        raise NotImplementedError


class NameMatchSelector(SinglePackSelector):
    r"""Select a :class:`DataPack` from a :class:`MultiPack` with specified
    name.
    """

    def __init__(self, 
                 configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__(configs)
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
        config = super().default_configs()
        config.update(
            {
                "select_name": None
            }
        )
        return config


class RegexNameMatchSelector(SinglePackSelector):
    r"""Select a :class:`DataPack` from a :class:`MultiPack` using a regex."""

    def __init__(self, 
                 configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__(configs)
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
        config = super().default_configs()
        config.update(
            {
                "select_name": None
            }
        )
        return config


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
