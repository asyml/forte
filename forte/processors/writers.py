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
from typing import Optional

from forte.common.exception import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base.writers import JsonPackWriter, MultiPackWriter


class PackIdJsonPackWriter(JsonPackWriter):
    def sub_output_path(self, pack: DataPack) -> Optional[str]:
        suffix = ".json.gz" if self.zip_pack else ".json"
        return str(pack.pack_id) + suffix


class PackNameJsonPackWriter(JsonPackWriter):
    def sub_output_path(self, pack: DataPack) -> Optional[str]:
        if pack.pack_name is None:
            raise ValueError(
                "Cannot use DocIdJsonPackWriter when [pack_name] of the pack "
                "is not set."
            )
        suffix = ".json.gz" if self.zip_pack else ".json"
        return pack.pack_name + suffix


class PackNameMultiPackWriter(MultiPackWriter):
    def pack_name(self, pack: DataPack) -> str:
        name = pack.pack_name
        if name is None:
            raise ProcessExecutionException(
                "Cannot used the DocIdMultiPackWriter because the [pack_name] "
                "is not assigned for the pack."
            )
        return name

    def multipack_name(self, pack: MultiPack) -> str:
        name = pack.pack_name
        if name is None:
            raise ProcessExecutionException(
                "Cannot used the DocIdMultiPackWriter because the doc id is "
                "not assigned for the pack."
            )
        return name


class PackIdMultiPackWriter(MultiPackWriter):
    def pack_name(self, pack: DataPack) -> str:
        return pack.pack_id

    def multipack_name(self, pack: MultiPack) -> str:
        return pack.pack_id
