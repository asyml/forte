# Copyright 2020 The Forte Authors. All Rights Reserved.
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
from typing import Iterable, Tuple
from abc import abstractmethod
from forte.data.caster import Caster
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack


__all__ = [
    "BaseDataAugmentProcessor"
]


class BaseDataAugmentProcessor(Caster[DataPack, MultiPack]):
    r"""The base class of processors that augments the data.
    Pass the input data_pack to the "cast" method
    and it will return an output Multipack.

    To inherit from this class, please override the augment function.
    """
    def cast(self, pack: DataPack) -> MultiPack:
        """
        Augment the data-pack into a multi-pack.

        Args:
            pack: The data pack to be augmented

        Returns: An iterator that produces the augmented multi pack.

        """
        p = MultiPack()
        p.add_pack_(pack, "original")
        augmented_data: Iterable[Tuple[str, DataPack]] = self.augment(pack)
        for pack_name, data_pack in augmented_data:
            p.add_pack_(data_pack, pack_name)
        return p

    @abstractmethod
    def augment(self, pack: DataPack) -> Iterable[Tuple[str, DataPack]]:
        r"""The method that augments the input datapack.
        It returns tuples of pack name and pack data.
        """
        raise NotImplementedError
