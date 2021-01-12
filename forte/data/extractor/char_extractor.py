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
from ft.onto.base_ontology import Annotation
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "CharExtractor"
]


class CharExtractor(BaseExtractor):
    r"""CharExtractor extracts feature from the text of entry.
    Text will be split into characters.

    Args:
        config: An instance of `Dict` or
            :class:`forte.common.configuration.Config` that provides all
            configurable options. See :meth:`default_configs` for available
            options and default values.
    """
    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        "max_char_length": int
            The maximum number of characters for one token in the text.
        """
        config = super().default_configs()
        config.update({"max_char_length": None})
        return config

    def update_vocab(self, pack: DataPack, instance: Annotation):
        r"""Add all character into vocabulary.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will get text from.
        """
        for word in pack.get(self.config.entry_type, instance):
            for char in word.text:
                self.add(char)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        r"""Extract the character feature of one instance.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        data = []
        max_char_length = -1

        for word in pack.get(self.config.entry_type, instance):
            if self.vocab:
                data.append([self.element2repr(char)
                    for char in word.text])
            else:
                data.append(list(word.text))
            max_char_length = max(max_char_length, len(data[-1]))

        if hasattr(self.config, "max_char_length") and \
            self.config.max_char_length is not None and \
            self.config.max_char_length < max_char_length:
            data = [token[:self.config.max_char_length] for
                    token in data]

        meta_data = {"need_pad": self.config.need_pad,
                     "pad_value": self.get_pad_value(),
                     "dim": 2,
                     "dtype": int if self.vocab else str}
        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)
