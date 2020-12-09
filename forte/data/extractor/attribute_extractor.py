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


from typing import Dict, Any, Union, Iterable
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor


class AttributeExtractor(BaseExtractor):
    '''This type of extractor will get the attribute on entry_type
    within one instance.
    '''
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)

        assert hasattr(self.config, "attribute_get"), \
            "Attribute is needed in AttributeExtractor"

        self.attribute_get = getattr(self.config, "attribute_get")

        # The default attribute_set is the same as attribute_get.
        if hasattr(self.config, "attribute_set"):
            self.attribute_set = getattr(self.config, "attribute_set")
        else:
            if isinstance(self.attribute_get, str):
                self.attribute_set = self.attribute_get
            else:
                raise AttributeError("Attribute_set need to be pass "
                        "in when attribute_get is not a field of str type.")

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            if callable(self.attribute_get):
                self.add(self.attribute_get(entry))
            else:
                self.add(getattr(entry, self.attribute_get))

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''The AttributeExtractor only extract one attribute for one entry
        in the instance. There for the output feature will have same number
        of attributes as entries in one instance.
        '''
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            if callable(self.attribute_get):
                attr = self.attribute_get(entry)
            else:
                attr = getattr(entry, self.attribute_get)
            if self.vocab:
                rep = self.element2repr(attr)
            else:
                rep = attr
            data.append(rep)
        # Data only has one dimension, therefore dim is 1.
        meta_data = {"pad_value": self.get_pad_id(),
                        "dim": 1,
                        "dtype": int if self.vocab else Any}
        return Feature(data=data,
                        metadata=meta_data,
                        vocab=self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Iterable[Union[int, Any]]):
        assert self.attribute_set != "text", "Text attribute is not"\
                                            "allowed to be set."
        instance_entry = list(pack.get(self.config.entry_type, instance))
        prediction = prediction[:len(instance_entry)]
        # TODO: we make some assumption here. The prediction is id.
        attrs = [self.id2element(int(x)) for x in prediction]
        for entry, attr in zip(instance_entry, attrs):
            if callable(self.attribute_set):
                self.attribute_set(entry, attr)
            else:
                setattr(entry, self.attribute_set, attr)
