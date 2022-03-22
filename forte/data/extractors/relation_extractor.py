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


from typing import Dict, Union, List, Optional, Tuple, Type

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.data.base_extractor import BaseExtractor
from forte.data.converter.feature import Feature
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation, Link
from forte.data.ontology.core import Entry

__all__ = ["LinkExtractor"]

from forte.utils import get_class


def get_index(
    pack: DataPack, index_entries: List[Annotation], context_entry: Annotation
):
    founds = []
    for i, entry in enumerate(index_entries):
        if pack.covers(context_entry, entry):
            founds.append(i)
    return [founds[0], founds[-1] + 1]


class LinkExtractor(BaseExtractor):
    """
    This extractor extracts relation type features from data packs. This
    extractor expects the parent and child of the relation to be Annotation
    entries.
    """

    def initialize(self, config: Union[Dict, Config]):
        # pylint: disable=attribute-defined-outside-init
        super().initialize(config)

        if self.config.attribute is None:
            raise ProcessorConfigError(
                "'attribute' is required in this extractor."
            )
        if self.config.index_annotation is None:
            raise ProcessorConfigError(
                "'index_annotation' is required in this extractor."
            )
        if self.config.entry_type is None:
            raise ProcessorConfigError(
                "'entry_type' is required in this extractor."
            )
        else:
            self._entry_class: Type[Link] = get_class(self.config.entry_type)

            if not issubclass(self._entry_class, Link):
                raise ProcessorConfigError(
                    "`entry_class` to this extractor " "must be a Link tpe."
                )

            self._parent_class: Type[Annotation] = self._entry_class.ParentType
            if not issubclass(self._parent_class, Annotation):
                raise ProcessorConfigError(
                    f"The parent class of the provided {self.config.entry_type}"
                    " must be an Annotation."
                )

            self._child_class: Type[Annotation] = self._entry_class.ChildType
            if not issubclass(self._child_class, Annotation):
                raise ProcessorConfigError(
                    f"The child class of the provided {self.config.entry_type}"
                    " must be an Annotation."
                )

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        Here:

        - "`entry_type`": The target relation entry type, should be a Link entry.
        - "`attribute`": The attribute of the relation to extract.
        - "`index_annotation`": The annotation object used to index the
          head and child node of the relations.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": None,
                "attribute": None,
                "index_annotation": None,
            }
        )
        return config

    def update_vocab(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        """
        Update values of relation attributes to the vocabulary.

        Args:
            pack (DataPack): The input data pack.
            context (Annotation): The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        Returns:
            None
        """

        entry: Entry
        for entry in pack.get(self.config.entry_type, context):
            attribute = getattr(entry, self.config.attribute)
            self.add(attribute)

    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        """Extract link data as features from the context.

        Args:
            pack (DataPack): The input data pack that contains the features.
            context (Annotation): The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns:

        """
        index_annotations: List[Annotation] = list(
            pack.get(self.config.index_annotation, context)
        )

        parent_nodes: List[Annotation] = []
        child_nodes: List[Annotation] = []
        relation_atts = []

        r: Link
        for r in pack.get(self.config.entry_type, context):
            parent_nodes.append(r.get_parent())  # type: ignore
            child_nodes.append(r.get_child())  # type: ignore

            raw_att = getattr(r, self.config.attribute)
            relation_atts.append(
                self.element2repr(raw_att) if self.vocab else raw_att
            )

        parent_unit_span = []
        child_unit_span = []

        for p, c in zip(parent_nodes, child_nodes):
            parent_unit_span.append(get_index(pack, index_annotations, p))
            child_unit_span.append(get_index(pack, index_annotations, c))

        meta_data = {
            "parent_unit_span": parent_unit_span,
            "child_unit_span": child_unit_span,
            "pad_value": self.get_pad_value(),
            "dim": 1,
            "dtype": int if self.vocab else str,
        }

        return Feature(data=relation_atts, metadata=meta_data, vocab=self.vocab)

    def add_to_pack(
        self,
        pack: DataPack,
        predictions: List[Tuple[Tuple[int, int], Tuple[int, int], int]],
        context: Optional[Annotation] = None,
    ):
        """
        Convert prediction back to Links inside the data pack.

        Args:
            pack (DataPack): The datapack to add predictions back.
            predictions (List): This is the output of the model, it is a
                triplet, the first element shows the parent, the second
                element shows the child. These two are indexed by the
                `index_annotation` of this extractor. The last element is the
                index of the relation attribute.
            context (Optional[Annotation]): The context is an Annotation
                entry where predictions will be added to. This has the same
                meaning with `context` as in
                :meth:`~forte.data.base_extractor.BaseExtractor.extract`.
                If None, then the whole data pack will be used as the
                context. Default is None.
        """
        index_entries: List[Annotation] = list(
            pack.get(self.config.index_annotation, context)
        )

        for parent, child, rel_index in predictions:
            parent_begin_entry_index, parent_end_entry_index = parent
            child_begin_entry_index, child_end_entry_index = child

            parent_start = index_entries[parent_begin_entry_index].begin
            parent_end = index_entries[parent_end_entry_index].end

            child_start = index_entries[child_begin_entry_index].begin
            child_end = index_entries[child_end_entry_index].end
            rel_value = self.id2element(rel_index)

            child_anno = self._child_class(
                pack, child_start, child_end  # type:ignore
            )

            parent_anno = self._parent_class(
                pack, parent_start, parent_end  # type:ignore
            )

            link = self._entry_class(
                pack, parent_anno, child_anno  # type:ignore
            )
            setattr(link, self.config.attribute, rel_value)
