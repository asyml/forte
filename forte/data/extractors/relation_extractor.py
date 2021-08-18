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


from typing import Dict, Any, Union, List
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.base_extractor import BaseExtractor
from forte.data.ontology import Annotation, Link
from forte.data.ontology.core import Entry

__all__ = ["LinkExtractor"]


def get_index(
    pack: DataPack, index_entries: List[Annotation], context_entry: Annotation
):
    founds = []
    for i, entry in enumerate(index_entries):
        if pack.covers(context_entry, entry):
            founds.append(i)
    return [founds[0], founds[-1] + 1]


class LinkExtractor(BaseExtractor):
    def initialize(self, config: Union[Dict, Config]):
        super().initialize(config)
        assert (
            self.config.attribute is not None
        ), "Configuration `attribute` should not be None."
        assert (
            self.config.based_on is not None
        ), "Configuration `based_on` should not be None."

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        "max_char_length": int
            The maximum number of characters for one token in the text.
        """
        config = super().default_configs()
        config.update({"attribute": None, "based_on": None})
        return config

    def update_vocab(self, pack: DataPack, instance: Annotation):
        entry: Entry
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            self.add(attribute)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        instance_based_on: List[Annotation] = list(
            pack.get(self.config.based_on, instance)
        )

        parent_nodes: List[Annotation] = []
        child_nodes: List[Annotation] = []
        relation_atts = []

        r: Link
        for r in pack.get(self.config.entry_type, instance):
            parent_nodes.append(r.get_parent())  # type: ignore
            child_nodes.append(r.get_child())  # type: ignore

            raw_att = getattr(r, self.config.attribute)
            relation_atts.append(
                self.element2repr(raw_att) if self.vocab else raw_att
            )

        parent_unit_span = []
        child_unit_span = []

        for p, c in zip(parent_nodes, child_nodes):
            parent_unit_span.append(get_index(pack, instance_based_on, p))
            child_unit_span.append(get_index(pack, instance_based_on, c))

        meta_data = {
            "parent_unit_span": parent_unit_span,
            "child_unit_span": child_unit_span,
            "pad_value": self.get_pad_value(),
            "dim": 1,
            "dtype": int if self.vocab else str,
        }

        return Feature(data=relation_atts, metadata=meta_data, vocab=self.vocab)

    def add_to_pack(
        self, pack: DataPack, instance: Annotation, prediction: Any
    ):
        # TODO: Unfinished
        pass
