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
This file implements BioSeqTaggingExtractor, which is used to extract feature
from the tagging label.
"""
import logging
from typing import Tuple, List, Dict, Union, Optional, Iterable, Type

from torch import Tensor
from forte.common.configuration import Config
from forte.data.converter.feature import Feature
from forte.data.data_pack import DataPack
from forte.data.base_extractor import BaseExtractor
from forte.data.extractors.utils import bio_tagging, add_entry_to_pack
from forte.data.ontology import Annotation
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = [
    "BioSeqTaggingExtractor"
]


class BioSeqTaggingExtractor(BaseExtractor):
    r"""BioSeqTaggingExtractor will the feature by performing BIO encoding
    for the attribute of entry and aligning to the tagging_unit entry. Most of
    the time, a user will not need to call this class explicitly, they will
    be called by the framework.

    Args:
        config: An instance of `Dict` or
            :class:`~forte.common.configuration.Config`.
            See :meth:`default_configs` for available options and
            default values.
    """

    def initialize(self, config: Union[Dict, Config]):
        # pylint: disable=attribute-defined-outside-init
        super().initialize(config=config)
        if self.config.attribute is None:
            raise AttributeError("attribute is required "
                                 "in BioSeqTaggingExtractor.")
        if not self.config.tagging_unit:
            raise AttributeError("tagging_unit is required in "
                                 "BioSeqTaggingExtractor.")
        self.attribute: str = self.config.attribute
        self.tagging_unit: Type[Annotation] = \
                get_class(self.config.tagging_unit)
        self.is_bert: bool = self.config.is_bert

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        Here:

        entry_type (str).
            Required. The string to the ontology type that the extractor
            will get feature from, e.g: `"ft.onto.base_ontology.EntityMention"`.

        attribute (str): Required. The attribute name of the
            entry from which labels are extracted.

        tagging_unit (str): Required. The tagging label
            will align to the tagging_unit Entry,
            e.g: `"ft.onto.base_ontology.Token"`.

        "vocab_method" (str)
            What type of vocabulary is used for this extractor.
            `raw`, `indexing`, `one-hot` are supported, default is `indexing`.
            Check the behavior of vocabulary under different setting
            in :class:`~forte.data.vocabulary.Vocabulary`

        "need_pad" (bool)
            Whether the `<PAD>` element should be added to vocabulary. And
            whether the feature need to be batched and padded. Default is True.
            When True, pad_value has to be set.

        "vocab_use_unk" (bool)
            Whether the `<UNK>` element should be added to vocabulary.
            Default is true.

        "pad_value" (int)
            ID assigned to pad. It should be integer smaller than 0.
            Default is 0.

        "vocab_unk_id" (int)
            ID assigned to unk. It should be integer smaller than 0.
            Default is 1.

        is_bert (bool)
            It indicates whether Bert model is used. If true, padding
            will be added to the beginning and end of a sentence
            corresponding to the special tokens ([CLS], [SEP])
            used in Bert. Default is False.

        For example, the config can be:

        .. code-block:: python

            {
                "entry_type": "ft.onto.base_ontology.EntityMention",
                "attribute": "ner_type",
                "tagging_unit": "ft.onto.base_ontology.Token"
            }

        The extractor will extract the BIO NER tags for instances.
        A possible feature can be:

        .. code-block:: python

            [[None, "O"], ["LOC", "B"], ["LOC", "I"], [None, "O"],
            [None, "O"], ["PER", "B"], [None, "O"]]

        """
        config = super().default_configs()
        config.update({"attribute": None,
                       "tagging_unit": "",
                       "is_bert": False})
        return config

    @classmethod
    def _bio_variance(cls, tag):
        r"""Return the BIO-schemed augmented tagging scheme, for example,
        if the `tag` is "person", the output would be `B-person`, `I-person`,
        `O-person`.

        Currently only supports B, I, O label.

        Args:
            tag (str): Tag name.
        """
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def predefined_vocab(self, predefined: Iterable):
        r"""Add predefined tags into the vocabulary. i.e. One can construct the
        tag vocabulary without exploring the training data.

        Args:
            predefined (Iterable[str]): A set of pre-defined tags.
        """
        for tag in predefined:
            for element in self._bio_variance(tag):
                self.add(element)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        r"""Add all the tag from one instance into the vocabulary.

        Args:
            pack (DataPack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.
        """
        for entry in pack.get(self._entry_type, instance):
            attribute = getattr(entry, self.attribute)
            for tag_variance in self._bio_variance(attribute):
                self.add(tag_variance)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        r"""Extract the sequence tagging feature of one instance. If the
        vocabulary of this extractor is set, then the extracted tag sequences
        will be converted to the tag ids (int).

        Args:
            pack (DataPack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns (Feature):
           a feature that contains the extracted data.
        """
        instance_tagged: List[Tuple[Optional[str], str]] = \
            bio_tagging(pack, instance,
            self.tagging_unit,
            self._entry_type,
            self.attribute)

        pad_value = self.get_pad_value()
        if self.vocab:
            # Use the vocabulary to map data into representation.
            vocab_mapped: List[Union[int, List[int]]] = []
            for pair in instance_tagged:
                vocab_mapped.append(self.element2repr(pair))
            raw_data: List = vocab_mapped
            if self.is_bert:
                raw_data = [pad_value] + raw_data + [pad_value]
        else:
            # When vocabulary is not available, use the original data.
            raw_data = instance_tagged

        meta_data = {"pad_value": pad_value,
                     "dim": 1,
                     "dtype": int if self.vocab else tuple}

        return Feature(data=raw_data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def pre_evaluation_action(self, pack: DataPack, instance: Annotation):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        By default, this function will remove tags in the instance. You can
        overwrite this function by yourself.

        Args:
            pack (DataPack): The datapack that contains the current
                instance.
            instance (Annotation): The instance on which the
                extractor performs the pre-evaluation action.
        """
        for entry in pack.get(self._entry_type, instance):
            pack.delete_entry(entry)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: List[int]):
        r"""Add the prediction for attribute to the instance. We make following
        assumptions for prediction.

            1. If we encounter "I" while its tag is different from the previous
               tag, we will consider this "I" as a "B" and start a new tag here.
            2. We will truncate the prediction it according to the number of
               entry. If the prediction contains `<PAD>` element, this should
               remove them.

        Args:
            pack (DataPack):
                The datapack that contains the current instance.
            instance (Annotation):
                The instance to which the extractor add prediction.
            prediction (Iterable[Union[int, Any]]):
                This is the output of the model, which contains the index for
                attributes of one instance.
        """
        instance_tagging_unit: List[Annotation] = \
            list(pack.get(self.tagging_unit, instance))

        if self.is_bert:
            prediction = prediction[1:-1]

        prediction = prediction[:len(instance_tagging_unit)]
        if isinstance(prediction, Tensor):
            prediction = prediction.cpu().numpy()
        tags = [self.id2element(x) for x in prediction]
        tag_start = None
        tag_end = None
        tag_type = None
        for entry, tag in zip(instance_tagging_unit, tags):
            if tag[1] == "O" or tag[1] == "B" or \
                    (tag[1] == "I" and tag[0] != tag_type):
                if tag_type:
                    entity_mention = add_entry_to_pack(pack,
                                                       self._entry_type,
                                                       tag_start,
                                                       tag_end)
                    setattr(entity_mention, self.attribute, tag_type)
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            else:
                tag_end = entry.end

        # Handle the final tag
        if tag_type and tag_start and tag_end:
            entity_mention = add_entry_to_pack(pack,
                                               self._entry_type,
                                               tag_start,
                                               tag_end)
            setattr(entity_mention, self.attribute, tag_type)
