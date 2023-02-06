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
from typing import (
    Any,
    Union,
    Iterable,
    Hashable,
    SupportsInt,
    Dict,
    Optional,
    List,
)

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.data.base_extractor import BaseExtractor
from forte.data.converter.feature import Feature
from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation

__all__ = ["AttributeExtractor"]


class AttributeExtractor(BaseExtractor):
    r"""`AttributeExtractor` extracts feature from the attribute of entry.
    Most of the time, a user will not need to call this class explicitly,
    they will be called by the framework.
    """

    def initialize(self, config: Union[Dict, Config]):
        super().initialize(config)
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        if self.config.entry_type is None:
            raise ProcessorConfigError(
                "The ``entry_type`` configuration must be "
                "provided and cannot be None."
            )
        if self.config.attribute is None:
            raise ProcessorConfigError(
                "The `attribute` configuration must be "
                "provided and cannot be None."
            )

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.


        Here:

        - "`attribute`": str
          The name of the attribute we want to extract from the entry. This
          attribute should present in the entry definition. There are some
          built-in attributes for some instance, such as `text` for
          `Annotation` entries. ``tid`` should be also available for any
          entries. The default value is ``tid``.
        - "`entry_type`": str
          The fully qualified name of the entry to extract attributes from.
          The default value is None, but this value must present or an
          `ProcessorConfigError` will be thrown.
        """
        config = super().default_configs()
        config.update(
            {
                "attribute": "tid",
                "entry_type": None,
            }
        )
        return config

    @classmethod
    def _get_attribute(cls, entry: Entry, attr: str) -> Any:
        r"""Get the attribute from entry. You can
        overwrite this function if you have special way to get the
        attribute from entry.

        Args:
            entry: An instance of Entry type, where the
                attribute will be extracted from.
            attr: The name of the attribute.

        Returns:
            Any. The attribute extracted from entry.
        """
        return getattr(entry, attr)

    @classmethod
    def _set_attribute(cls, entry: Entry, attr: str, value: Any):
        r"""Set the attribute of an entry to value.
        You can overwrite this function if you have special way to
        set the attribute.

        Args:
            entry: An instance of Entry type, where the
                attribute will be set.
            attr: The name of the attribute.
            value: The value to be set for the attribute.
        """
        if attr == "text":
            raise AttributeError("text attribute of entry cannot be changed.")
        setattr(entry, attr, value)

    def update_vocab(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        r"""Get all attributes of one instance and add them into the vocabulary.

        Args:
            pack: The data pack input to extract vocabulary.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        entry: Entry
        for entry in pack.get(self.config.entry_type, context):
            # The following pylint skip due to a bug:
            # https://github.com/PyCQA/pylint/issues/3507
            # Hashable is not recognized the type.
            # pylint: disable=isinstance-second-argument-not-valid-type
            element = self._get_attribute(entry, self.config.attribute)
            if not isinstance(element, Hashable):
                raise AttributeError(
                    "Only hashable element can be"
                    "added into the vocabulary. Consider setting"
                    "vocab_method to be raw and do not call update_vocab"
                    "if you only need the raw attribute value without"
                    "converting them into index."
                )
            self.add(element)

    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        """Extract the attribute of an entry of the configured entry type.
        The entry type is passed in from via extractor config `entry_type`.

        Args:
            pack: The datapack that contains the current instance.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns:
            Features (attributes) for instance with in the provided
            context, they will be converted to the representation based on
            the vocabulary configuration.
        """
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        data = []

        instance: Annotation
        for instance in pack.get(self.config.entry_type, context):
            value = self._get_attribute(instance, self.config.attribute)
            rep = self.element2repr(value) if self.vocab else value
            data.append(rep)

        meta_data = {
            "need_pad": self.config.need_pad,
            "pad_value": self.get_pad_value(),
            "dim": 1,
            "dtype": int if self.vocab else Any,
        }
        return Feature(data=data, metadata=meta_data, vocab=self.vocab)

    def pre_evaluation_action(
        self, pack: DataPack, context: Optional[Annotation]
    ):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        By default, this function will remove all attributes defined in the
        config (set them to None). You can overwrite this function by yourself.

        Args:
            pack: The datapack that contains the current
                instance.
            context: The context is an Annotation entry where
                data are extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        entry: Entry
        for entry in pack.get(self.config.entry_type, context):
            self._set_attribute(entry, self.config.attribute, None)

    def add_to_pack(
        self,
        pack: DataPack,
        predictions: Iterable[SupportsInt],
        context: Optional[Annotation] = None,
    ):
        r"""Add the prediction for attributes to the data pack.
        We assume the number of predictions in the iterable to be the same as
        the number of the entries of the defined type in the data pack.

        Args:
            pack: The datapack that contains the current
                instance.
            predictions: This is the output
                of the model, which should be the class index for the attribute.
            context: The context is an Annotation
                entry where predictions will be added to. This has the same
                meaning with `context` as in
                :meth:`~forte.data.base_extractor.BaseExtractor.extract`.
                If None, then the whole data pack will be used as the
                context. Default is None.
        """
        if self.config is None:
            raise ProcessorConfigError(
                "Configuration for the extractor not found."
            )

        instance_entries: List[Entry] = list(
            pack.get(self.config.entry_type, context)
        )

        # The following pylint skip due to a bug:
        # https://github.com/PyCQA/pylint/issues/3507
        # Iterable is not recognized the type.
        # pylint: disable=isinstance-second-argument-not-valid-type
        # _predictions = predictions if isinstance(predictions, Iterable) else \
        #     [predictions]
        # if not isinstance(predictions, Iterable):
        #     predictions = [predictions]
        values = [self.id2element(int(x)) for x in predictions]
        for entry, value in zip(instance_entries, values):
            self._set_attribute(entry, self.config.attribute, value)
