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


from typing import Dict, Any, Union
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack, DataIndex
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor


class LinkExtractor(BaseExtractor):
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
            "based_on": None
        }
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)
        assert self.config.attribute is not None, \
            "Attribute should not be None."
        assert self.config.based_on is not None, \
            "Based_on should not be None."

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            self.add(attribute)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        instance_based_on = list(pack.get(self.config.based_on, instance))
        instance_entry = list(pack.get(self.config.entry_type, instance))
        parent_entry = [entry.get_parent() for entry in instance_entry]
        child_entry = [entry.get_child() for entry in instance_entry]

        data = [getattr(entry, self.config.attribute)
                    for entry in instance_entry]

        if self.vocab:
            data = [self.element2repr(entry) for entry in data]

        parent_unit_span = []
        child_unit_span = []

        for p, c in zip(parent_entry, child_entry):
            parent_unit_span.append(self.get_index(instance_based_on, p))
            child_unit_span.append(self.get_index(instance_based_on, c))

        meta_data = {
            "parent_unit_span": parent_unit_span,
            "child_unit_span": child_unit_span,
            "pad_value": self.get_pad_id(),
            "dim": 1,
            "dtype": int if self.vocab else str
        }

        return Feature(data = data,
                        metadata = meta_data,
                        vocab = self.vocab)

    def get_index(self, inner_entries, span):
        index = DataIndex()
        founds = []
        for i, entry in enumerate(inner_entries):
            if index.in_span(entry, span):
                founds.append(i)
        return [founds[0], founds[-1]+1]

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        pass
