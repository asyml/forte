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
from typing import Dict, Any
from abc import abstractmethod, ABC
from forte.data.ontology.core import Entry

__all__ = [
    "BaseDataAugmenter",
    "ReplacementDataAugmenter"
]


class BaseDataAugmenter(ABC):
    r"""
    The base class holds the data augmentation algorithm.
    We leave the :func: augment method to be implemented
    by subclasses because the function signature might be
    different.
    """
    def __init__(self, configs: Dict[str, Any]):
        r"""
        Set the configuration for the data augmenter.
        """
        self.configs = configs

    @abstractmethod
    def augment(self, input: Entry) -> str:
        r"""
        This function takes in an entry as input and
        returns the augmented string.
        """
        raise NotImplementedError


class ReplacementDataAugmenter(BaseDataAugmenter):
    r"""
    Most data augmentation algorithms can be considered as replacement-based
    methods on different levels.
    """
