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
Processors that process pack.
"""
from abc import ABC

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
    "MultiPackProcessor"
]


class BasePackProcessor(BaseProcessor[PackType], ABC):
    r"""The base class of processors that process one pack sequentially. If you
    are looking for batching (that might happen across packs, refer to
    :class:`BaseBatchProcessor`.
    """
    pass


class PackProcessor(BaseProcessor[DataPack], ABC):
    r"""The base class of processors that process one :class:`DataPack` each
    time.
    """

    def _process(self, input_pack: DataPack):
        raise NotImplementedError


class MultiPackProcessor(BaseProcessor[MultiPack], ABC):
    r"""The base class of processors that process :class:`MultiPack` each time.
    """

    def _process(self, input_pack: MultiPack):
        raise NotImplementedError
