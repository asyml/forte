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
This file contains examples of PackProcessor implementations, the processors
here are useful as placeholders and test cases.
"""

from forte.data import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "DummyPackProcessor",
]


class DummyPackProcessor(PackProcessor):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super().__init__()

    def _process(self, input_pack: DataPack):
        pass
