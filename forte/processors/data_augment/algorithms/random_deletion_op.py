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
from typing import Tuple
from ft.onto.base_ontology import Entry
from forte.utils.utils import create_class_with_kwargs
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp

__all__ = [
    "RandomDeletionOp",
]


class RandomDeletionOp(TextReplacementOp):
    r"""
    This class is a replacement op that randomly deletes words by
    replacing the word with an empty string.

    """
    def __init__(self, configs: Config):
        super().__init__(configs)

    def replace(self, token: Entry) -> Tuple[bool, str]:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        Args:
            - token: The input entry.
        Returns:
            - A boolean value, True if the replacement happens.
            - A synonym of the word.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs["prob"]:
            return False, token.text
        return True, ""
