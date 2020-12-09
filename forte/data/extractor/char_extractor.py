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


from typing import Dict, Union
from ft.onto.base_ontology import Token, Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor


class CharExtractor(BaseExtractor):
    '''CharExtractor will get each char for each token in the instance.'''
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        assert hasattr(self.config, "entry_type") and \
            getattr(self.config, "entry_type") == Token, \
            """CharExtractor is only used to extract characters of Token.
            The entry_type can only be Token."""

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for word in pack.get(self.config.entry_type, instance):
            for char in word.text:
                self.add(char)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
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
            self.config.max_char_length < max_char_length:
            data = [token[:self.config.max_char_length] for
                    token in data]

        # Data has two dimensions, therefore dim is 2.
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 2,
                    "dtype": int if self.vocab else str}
        return Feature(data=data, metadata=meta_data,
                        vocab=self.vocab)
