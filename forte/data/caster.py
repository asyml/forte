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
This defines the switcher interface, which can cast the pack types in the
middle of a pipeline flow. The main difference of this from Selector is that
the returned pack will be used to replaced the original pack, while in Selector,
the original pack is kept unchanged.
"""
from abc import ABC
from typing import Generic, TypeVar

from forte.data.base_pack import BasePack
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.pipeline_component import PipelineComponent

InputPackType = TypeVar('InputPackType', bound=BasePack)
OutputPackType = TypeVar('OutputPackType', bound=BasePack)


class Caster(PipelineComponent[InputPackType],
             Generic[InputPackType, OutputPackType], ABC):
    def cast(self, pack: InputPackType) -> OutputPackType:
        raise NotImplementedError


class MultiPackBoxer(Caster[DataPack, MultiPack]):
    """
    This class creates a MultiPack from a DataPack, this MultiPack will only
    contains the original DataPack, indexed by the :attr:`pack_name`.
    """

    def cast(self, pack: DataPack) -> MultiPack:
        """
        Args:
            pack: The data pack to be boxed

        Returns: An iterator that produces the boxed multi pack.

        """
        p = MultiPack(self._pack_manager)
        p.add_pack_(pack, self.configs.pack_name)
        return p

    @classmethod
    def default_configs(cls):
        return {
            'pack_name': 'default'
        }
