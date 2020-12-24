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

import logging
from typing import Dict, Any, Union, Iterable
from collections import abc
from ft.onto.base_ontology import Entry, Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "AttributeExtractor"
]


class AttributeExtractor(BaseExtractor):
    r"""AttributeExtractor extracts feature from the attribute of entry.

    Args:
        config: An instance of `Dict` or
            :class:`forte.common.configuration.Config`

            attribute: str. Required. The attribute name of the
                entry from which features will be extracted. For
                example, "text" attribute of Token.
    """
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)

        if "attribute" not in self.config:
            raise AttributeError("attribute needs to be specified in "
                                "the configuration of an AttributeExtractor.")

    @property
    def state(self) -> Dict:
        state = super().state
        state.update({
            "attribute": self.config.attribute
        })
        return state

    @classmethod
    def from_state(cls, state: Dict) -> "AttributeExtractor":
        obj = super().from_state(state)
        additional = {
            "attribute": state["attribute"]
        }
        obj.config = Config(obj.config, additional,
                            allow_new_hparam=True)
        obj = cls(obj)
        return obj

    @staticmethod
    def get_attribute(entry: Entry, attr: str) -> Any:
        r"""Functionality: Get the attribute from entry. You can
        overwrite this function if you have sepcial way to get the
        attribute from entry.
        """
        return getattr(entry, attr)

    @staticmethod
    def set_attribute(entry: Entry, attr: str, value: Any):
        r"""Functionality: Set the attribute of an entry to value.
        You can overwrite this function if you have sepcial way to
        set the attribute.
        """
        if attr == "text":
            raise AttributeError("text attribute of entry cannot "
                                "be changed.")
        setattr(entry, attr, value)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        r"""Functionality: Get all attributes of one instance and
        add them into the vocabulary.
        """
        for entry in pack.get(self.config.entry_type, instance):
            self.add(self.get_attribute(entry, self.config.attribute))

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        r"""Functionality: Extractor the attributes of one instance.
        For example, the text of tokens in one sentence.
        """
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            value = self.get_attribute(entry, self.config.attribute)
            rep = self.element2repr(value) if self.vocab else value
            data.append(rep)

        meta_data = {"pad_value": self.get_pad_value(),
                     "dim": 1,
                     "dtype": int if self.vocab else Any}

        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def remove_from_pack(self, pack: DataPack, instance: Annotation):
        r"""Funcationality: Remove attributes of one instance. For
        example remove all pos tags of tokens in one sentence, if the
        entry_type is Token and the attribute is pos.
        """
        for entry in pack.get(self.config.entry_type, instance):
            self.set_attribute(entry, self.config.attribute, None)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Iterable[Union[int, Any]]):
        r"""Functionality: Add the prediction for attribute to the
        instance. If the prediction is an iterable object, we assume
        each of the element in prediction will correspond to one entry.
        If the prediction is only one element, then we assume there will
        only be one entry in the instance."""
        instance_entry = list(pack.get(self.config.entry_type, instance))

        if not isinstance(prediction, abc.Iterable):
            prediction = [prediction]
        values = [self.id2element(int(x)) for x in prediction]
        for entry, value in zip(instance_entry, values):
            self.set_attribute(entry, self.config.attribute, value)
