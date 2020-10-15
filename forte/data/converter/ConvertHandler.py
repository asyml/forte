#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import List, Dict

from forte.data.ontology.core import EntryType
from torch import Tensor

from forte.data.base_pack import BasePack


class ConvertHandler():
    def __init__(self, resource: Dict):
        self.resource = resource

    def convert(self, data_packs: List[BasePack]) -> Dict[str, Tensor]:
        scope: EntryType = self.resource["scope"]
        schemes: Dict = self.resource["schemes"]
        tensor_collection: Dict = {}

        for data_pack in data_packs:
            for instance in data_pack.get(scope):
                for tag, scheme in schemes.items():
                    scheme["converter"].consume_instance(data_pack, instance)

        for tag, scheme in schemes.items():
            tensor_collection[tag] = scheme["converter"].produce_tensor()

        return tensor_collection
