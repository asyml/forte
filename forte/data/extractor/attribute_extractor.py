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
This file implements AttributeExtractor, which is used to extract feature
from the attribute of entries.
"""
import logging
from typing import Any, Union, Iterable, Hashable
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "AttributeExtractor"
]


class AttributeExtractor(BaseExtractor):
    r"""AttributeExtractor extracts feature from the attribute of entry.
    Most of the time, a user will not need to call this class explicitly,
    they will be called by the framework.

    Args:
        config: An instance of `Dict` or
            :class:`forte.common.configuration.Config`

            attribute (str): Required. The attribute name of the
                entry from which features will be extracted. For
                example, "text" attribute of Token.
    """

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        "attribute": str
            The name of attribute we want to extract from the entry.
        """
        config = super().default_configs()
        config.update({"attribute": "text"})
        return config

    def get_attribute(self, entry: Entry, attr: str) -> Any:
        r"""Get the attribute from entry. You can
        overwrite this function if you have special way to get the
        attribute from entry.

        Args:
            entry (Entry): An instance of Entry type, where the
                attribute will be extracted from.
            attr (str): The name of the attribute.

        Returns:
            Any. The attribute extracted from entry.
        """
        return getattr(entry, attr)

    def set_attribute(self, entry: Entry, attr: str, value: Any):
        r"""Set the attribute of an entry to value.
        You can overwrite this function if you have special way to
        set the attribute.

        Args:
            entry (Entry): An instance of Entry type, where the
                attribute will be set.
            attr (str): The name of the attribute.
            value (Any): The value to be set for the attribute.
        """
        if attr == "text":
            raise AttributeError("text attribute of entry cannot "
                                 "be changed.")
        setattr(entry, attr, value)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        r"""Get all attributes of one instance and
        add them into the vocabulary.

        Args:
            pack (DataPack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.
        """
        for entry in pack.get(self.config.entry_type, instance):
            element = self.get_attribute(entry, self.config.attribute)
            if not isinstance(element, Hashable):
                raise AttributeError(
                    "Only hashable element can be"
                    "added into the vocabulary. Consider setting"
                    "vocab_method to be raw and do not call update_vocab"
                    "if you only need the raw attribute value without"
                    "coverting them into index.")
            self.add(element)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        r"""Extract attributes of one instance.
        For example, the text of tokens in one sentence.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            value = self.get_attribute(entry, self.config.attribute)
            rep = self.element2repr(value) if self.vocab else value
            data.append(rep)

        meta_data = {"need_pad": self.config.need_pad,
                     "pad_value": self.get_pad_value(),
                     "dim": 1,
                     "dtype": int if self.vocab else Any}

        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def pre_evaluation_action(self, pack: DataPack, instance: Annotation):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        By default, this function will remove the attribute. You can
        overwrite this function by yourself.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.
        """
        for entry in pack.get(self.config.entry_type, instance):
            self.set_attribute(entry, self.config.attribute, None)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Iterable[Union[int, Any]]):
        r"""Add the prediction for attribute to the
        instance. If the prediction is an iterable object, we assume
        each of the element in prediction will correspond to one entry.
        If the prediction is only one element, then we assume there will
        only be one entry in the instance.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance to which the
                extractor add prediction.
            prediction (Iterable[Union[int, Any]]): This is the output
                of the model, which contains the index for attributes
                of one instance.
        """
        instance_entry = list(pack.get(self.config.entry_type, instance))

        if not isinstance(prediction, Iterable):
            prediction = [prediction]
        values = [self.id2element(int(x)) for x in prediction]
        for entry, value in zip(instance_entry, values):
            self.set_attribute(entry, self.config.attribute, value)
