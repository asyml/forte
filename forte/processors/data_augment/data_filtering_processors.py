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
import random

from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor


"""
Processors that filter the augmented data. The processor filter
augmented data based on metrics such as domain-relevance.
"""
class RandomDataFilteringProcessor(MultiPackProcessor):
    r"""Data filtering processor that filters augmented data by
    randomly selecting examples.
    """
    def __init__(self, configs: Config):
        self.model = lambda data_pack : random.uniform(0, 1)
        self.configs = Config(configs, self.default_configs())

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        configs.update({
            "threshold": 0.5
        })
        return configs

    def _process(self, input_pack: MultiPack):
        for pack_name in input_pack.pack_names:
            input_pack = input_pack.get_pack(pack_name)
            if model(input_pack) < self.configs.threshold:
                # currently MultiPack has no remove_pack
                input_pack.remove_pack(pack_name)
