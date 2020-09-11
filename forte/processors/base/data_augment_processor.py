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
Processors that augment the data.
"""
from abc import abstractmethod
from forte.data.caster import Caster
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack


__all__ = [
    "BaseDataAugmentProcessor"
]


class BaseDataAugmentProcessor(Caster[DataPack, MultiPack]):
    r"""The base class of processors that augments the data.
    """
    @abstractmethod
    def cast(self, pack: DataPack) -> MultiPack:
        """
        Augment the data-pack into a multi-pack.

        Args:
            pack: The data pack to be augmented

        Returns: An iterator that produces the augmented multi pack.

        """
        p = MultiPack()
        p.add_pack_(pack, self.configs.pack_name)
        return p

    @classmethod
    def default_configs(cls):
        return {
            'pack_name': 'default'
        }
