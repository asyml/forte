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
from forte.data.base_extractor import BaseExtractor
from forte.data.converter.feature import Feature
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.ontology.core import Entry
from forte.datasets.conll.conll_utils import bio_tagging
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = ["BioSeqTaggingExtractor"]


class BioSeqTaggingExtractor(BaseExtractor):
    r"""BioSeqTaggingExtractor will the feature by performing BIO encoding
    for the attribute of entry and aligning to the tagging_unit entry. Most of
    the time, a user will not need to call this class explicitly, they will
    be called by the framework.
    """

    def initialize(self, config: Union[Dict, Config]):
        """
        Initialize the extractor based on the provided configuration.

        Args:
            config: The configuration of the extractor, it can be a `Dict` or
                :class:`~forte.common.configuration.Config`.
                See :meth:`default_configs` for available options and
                default values.
        """
        # pylint: disable=attribute-defined-outside-init
        super().initialize(config=config)
        if self.config.attribute is None:
            raise AttributeError(
                "attribute is required " "in BioSeqTaggingExtractor."
            )
        if not self.config.tagging_unit:
            raise AttributeError(
                "tagging_unit is required in " "BioSeqTaggingExtractor."
            )
        self._attribute: str = self.config.attribute
        self._tagging_unit: Type[Annotation] = get_class(
            self.config.tagging_unit
        )
        self._entry_type: Type[Annotation] = get_class(self.config.entry_type)

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        Here, additional parameters are added from the parent class:

        - entry_type (str): Required. The fully qualified name of an
          Annotation entry to extract attribute from. For example,
          for an NER task, it could be `ft.onto.base_ontology.EntityMention`.

        - attribute (str): Required. The attribute name of the
          entry from which labels are extracted.

        - tagging_unit (str): Required. The fully qualified name of the
          units for tagging, The tagging label will align to the units,
          e.g: `ft.onto.base_ontology.Token`.

        - pad_value (int):
          A customized value/representation to be used for
          padding. This value is only needed when `use_pad` is True.
          Default is -100 to follow PyTorch convention.

        - is_bert (bool):
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
        config.update(
            {
                "entry_type": None,
                "attribute": None,
                "tagging_unit": "",
                "pad_value": -100,
                "is_bert": False,
            }
        )
        return config

    @classmethod
    def _bio_variance(cls, tag: str):
        r"""Return the BIO-schemed augmented tagging scheme, for example,
        if the `tag` is "person", the output would be `B-person`, `I-person`,
        `O-person`.

        Currently only supports B, I, O label.

        Args:
            tag: Tag name.
        """
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def predefined_vocab(self, predefined: Iterable):
        r"""Add predefined tags into the vocabulary. i.e. One can construct the
        tag vocabulary without exploring the training data.

        Args:
            predefined: A set of pre-defined tags.
        """
        for tag in predefined:
            for element in self._bio_variance(tag):
                self.add(element)

    def update_vocab(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        r"""Add all the tag from one instance into the vocabulary.

        Args:
            pack: The datapack that contains the current
                instance.
            context: The context is an Annotation entry
                where features will be extracted within its range. If None,
                then the whole data pack will be used as the context.
                Default is None.

        """
        anno: Annotation
        for anno in pack.get(self.config.entry_type, context):
            for tag_variance in self._bio_variance(
                getattr(anno, self._attribute)
            ):
                self.add(tag_variance)

    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        r"""Extract the sequence tagging feature of one instance. If the
        vocabulary of this extractor is set, then the extracted tag sequences
        will be converted to the tag ids (int).

        Args:
            pack: The datapack that contains the current
                instance.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns (Feature): a feature that contains the extracted BIO sequence
            of and other metadata.
        """
        instance_tagged: List[Tuple[Optional[str], str]] = bio_tagging(
            pack,
            self.config.tagging_unit,
            self.config.entry_type,
            self.config.attribute,
            context,
        )

        pad_value = self.get_pad_value()
        if self.vocab:
            # Use the vocabulary to map data into representation.
            vocab_mapped: List[Union[int, List[int]]] = []
            for pair in instance_tagged:
                vocab_mapped.append(self.element2repr(pair))
            raw_data: List = vocab_mapped
            if self.config.is_bert:
                raw_data = [pad_value] + raw_data + [pad_value]

            need_pad = self.vocab.use_pad
        else:
            # When vocabulary is not available, use the original data.
            raw_data = instance_tagged
            need_pad = self.config.need_pad

        meta_data = {
            "need_pad": need_pad,
            "pad_value": pad_value,
            "dim": 1,
            "dtype": int if self.vocab else tuple,
        }
        return Feature(data=raw_data, metadata=meta_data, vocab=self.vocab)

    def pre_evaluation_action(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        By default, this function will remove tags in the instance. You can
        overwrite this function by yourself.

        Args:
            pack: The datapack to be processed.
            context: The context is an Annotation entry where
                data are extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        all_entries: List[Entry] = []
        entry: Entry
        for entry in pack.get(self.config.entry_type, context):
            all_entries.append(entry)

        for e in all_entries:
            pack.delete_entry(e)

    def add_to_pack(
        self,
        pack: DataPack,
        predictions: List[int],
        context: Optional[Annotation] = None,
    ):
        r"""Add the prediction results to data pack. The predictions are

        We make following assumptions for prediction.

            1. If we encounter "I" while its tag is different from the previous
               tag, we will consider this "I" as a "B" and start a new tag here.
            2. We will truncate the prediction it according to the number of
               entry. If the prediction contains `<PAD>` element, this should
               remove them.

        Args:
            pack: The datapack that contains the current instance.
            predictions:
                This is the output of the model, which contains the index for
                attributes of one instance.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        instance_tagging_unit: List[Annotation] = list(
            pack.get(self._tagging_unit, context)
        )

        if self.config.is_bert:
            predictions = predictions[1:-1]

        predictions = predictions[: len(instance_tagging_unit)]
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()

        tags = [self.id2element(x) for x in predictions]
        tag_start = None
        tag_end = None
        tag_type = None
        for entry, tag in zip(instance_tagging_unit, tags):
            if (
                tag[1] == "O"
                or tag[1] == "B"
                or (tag[1] == "I" and tag[0] != tag_type)
            ):
                if tag_type:
                    entity_mention = self._entry_type(pack, tag_start, tag_end)
                    setattr(entity_mention, self._attribute, tag_type)
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            else:
                tag_end = entry.end

        # Handle the final tag
        if tag_type and tag_start and tag_end:
            entity_mention = self._entry_type(
                pack, tag_start, tag_end  # type: ignore
            )
            setattr(entity_mention, self._attribute, tag_type)
