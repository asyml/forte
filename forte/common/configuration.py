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
We are going to reuse the configuration class HParams in Texar. However, the
name might be confusing in the context of processors, so we rename them to
Config here.
"""
from typing import Dict

from texar.torch import HParams

Config = HParams


def merge_configs(config: Dict, parent_config: Dict) -> Dict:
    return HParams(config, parent_config, True).todict()
