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
Class for data augmentation algorithm. The text replacement op will
replace a piece of text with data augmentation algorithms.
"""
from typing import Tuple
from abc import abstractmethod, ABC
from forte.data.ontology.top import Annotation
from forte.common.configuration import Config

__all__ = [
    "TextReplacementOp",
]


class TextReplacementOp(ABC):
    r"""
    The base class holds the data augmentation algorithm.
    We leave the :func: replace method to be implemented
    by subclasses.
    """
    def __init__(self, configs: Config):
        r"""
        Set the configuration for the text replacement op.
        """
        self.configs = configs

    @abstractmethod
    def replace(self, input: Annotation) -> Tuple[bool, str]:
        r"""
        Most data augmentation algorithms can be considered
        as replacement-based methods on different levels.

        Args:
            - input: the input annotation to be replaced.
        Returns:
            - A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        raise NotImplementedError
