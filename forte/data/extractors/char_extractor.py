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
This file implements CharExtractor, which is used to extract feature
from characters of a piece of text.
"""
import logging
from typing import Optional

from forte.common import ProcessorConfigError
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.base_extractor import BaseExtractor
from forte.data.ontology import Annotation

logger = logging.getLogger(__name__)

__all__ = ["CharExtractor"]


class CharExtractor(BaseExtractor):
    r"""CharExtractor extracts feature from the text of entry.
    Text will be split into characters.
    """

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default configuration parameters.

        Here:

        - "max_char_length": int
          The maximum number of characters for one token in the text,
          default is None, which means no limit will be set.
        - "entry_type": str
          The fully qualified name of an annotation type entry. Characters
          will be extracted based on these entries. Default is `Token`,
          which means characters of tokens will be extracted.
        """
        config = super().default_configs()
        config.update(
            {
                "max_char_length": None,
                "entry_type": "ft.onto.base_ontology.Token",
            }
        )
        return config

    def update_vocab(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        r"""Add all character into vocabulary.

        Args:
            pack: The input data pack.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        word: Annotation
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )
        for word in pack.get(self.config.entry_type, context):
            for char in word.text:  # type: ignore
                self.add(char)

    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        r"""Extract the character feature of one instance.

        Args:
            pack: The datapack to extract features from.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns:
            a iterator of feature that contains the characters of each
            specified annotation.
        """
        data = []

        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        entry: Annotation
        for entry in pack.get(self.config.entry_type, context):
            if self.config.max_char_length is not None:
                max_char_length = min(
                    self.config.max_char_length, len(entry.text)  # type: ignore
                )
            else:
                max_char_length = len(entry.text)  # type: ignore

            characters = entry.text[:max_char_length]  # type: ignore

            if self.vocab:
                data.append([self.element2repr(char) for char in characters])
            else:
                data.append(list(characters))

        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        meta_data = {
            "need_pad": self.config.need_pad,
            "pad_value": self.get_pad_value(),
            "dim": 2,
            "dtype": int if self.vocab else str,
        }
        return Feature(data=data, metadata=meta_data, vocab=self.vocab)
