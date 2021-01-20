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
from typing import Tuple, List, Dict, Union, Optional
from ft.onto.base_ontology import Annotation, EntityMention
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.utils import bio_tagging
from forte.data.extractor.base_extractor import BaseExtractor

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
            :class:`forte.common.configuration.Config`.
            See :meth:`default_configs` for available options and
            default values.
    """
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)

        if self.config.attribute is None:
            raise AttributeError("attribute is required "
                            "in BioSeqTaggingExtractor.")
        if self.config.tagging_unit is None:
            raise AttributeError("tagging_unit is required in "
                                "BioSeqTaggingExtractor.")

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        entry_type: Type[Entry]. Required. The ontology type that the
            extractor will get feature from.

        attribute (str): Required. The attribute name of the
            entry from which labels are extracted.

        tagging_unit (Type[Entry]): Required. The tagging label
            will align to the tagging_unit Entry.

        For example, the config can be "entry_type": EntityMention,
            "attribute": "ner_type", "tagging_unit": Token.

        The extractor will extract the BIO NER tags for instances.
            A possible feature can be [[None, "O"], [LOC, "B"], [LOC, "I"],
            [None, "O"], [None, "O"], [PER, "B"], [None, "O"]]
        """
        config = super().default_configs()
        config.update({"attribute": None,
                        "tagging_unit": None})
        return config

    def _bio_variance(self, tag):
        r"""Return the B, I, O label with tag.

        Args:
            tag (str): Tag name.
        """
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def predefined_vocab(self, predefined: set):
        r"""Add predefined tags into the vocabulary.

        Args:
            predefined (set): A set of tags.
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
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            for tag_variance in self._bio_variance(attribute):
                self.add(tag_variance)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        r"""Extract the sequence tagging feature of one instance.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        instance_tagged: List[Tuple[Optional[str], str]] = \
            bio_tagging(pack, instance,
            self.config.tagging_unit, self.config.entry_type,
            self.config.attribute)

        data = []
        for pair in instance_tagged:
            if self.vocab:
                data.append(self.element2repr(pair))
            else:
                data.append(pair)
        meta_data = {"pad_value": self.get_pad_value(),
                     "dim": 1,
                     "dtype": int if self.vocab else tuple}
        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def pre_evaluation_action(self, pack: DataPack, instance: Annotation):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        By default, this function will remove tags in the instance. You can
        overwrite this function by yourself.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance on which the
                extractor performs the pre-evaluation action.
        """
        for entry in pack.get(self.config.entry_type, instance):
            pack.delete_entry(entry)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: List[int]):
        r"""Add the prediction for attribute to the instance.
        We make following assumptions for prediction.
        1. If we encounter "I" while its tag is different from the previous tag,
        we will consider this "I" as a "B" and start a new tag here.
        2. We will truncate the prediction it according to the number of entry.
        If the prediction contains <PAD> element, this should remove them.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance to which the
                extractor add prediction.
            prediction (Iterable[Union[int, Any]]): This is the output
                of the model, which contains the index for attributes
                of one instance.
        """
        instance_tagging_unit: List[Annotation] = \
            list(pack.get(self.config.tagging_unit, instance))
        prediction = prediction[:len(instance_tagging_unit)]
        tags = [self.id2element(x) for x in prediction]
        tag_start = None
        tag_end = None
        tag_type = None
        for entry, tag in zip(instance_tagging_unit, tags):
            if tag[1] == "O" or tag[1] == "B" or \
                (tag[1] == "I" and tag[0] != tag_type):
                if tag_type:
                    entity_mention = EntityMention(pack, tag_start, tag_end)
                    entity_mention.ner_type = tag_type
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            else:
                tag_end = entry.end

        # Handle the final tag
        if tag_type is not None and \
            tag_start is not None and \
            tag_end is not None:
            entity_mention = EntityMention(pack, tag_start, tag_end)
            entity_mention.ner_type = tag_type
