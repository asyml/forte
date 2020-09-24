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
Class for data augmentation algorithm.
"""
from typing import List, Dict
from abc import abstractmethod, ABC


__all__ = [
    "BaseDataAugmenter"
]

class BaseDataAugmenter(ABC):
    r"""
    The base class holds the data augmentation algorithm.
    All algorithms can be considered as replacement-based methods
    on different levels(character/word/sentence). The replacement_level
    is a list containing the levels it allows.

    For example, the replacement_level of synonym replacement is ["word"],
    that of back-translation is ["word", "sentence"].
    """
    @property
    @abstractmethod
    def replacement_level(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def augment(self, input: str, additional_info: Dict[str, str]) -> str:
        raise NotImplementedError
