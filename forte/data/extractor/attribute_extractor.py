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
    """AttributeExtractor will get feature from the attribute on entry.
    Args:
        config:
            Required keys:
            "attribute_get": str or Callable. If str, the extracted
                feature comes from calling `getattr(entry, attribute_get)`.
                If Callable, the extracted feature comes from
                `attribute_get(entry)`.

            Optional keys:
            "attribute_set": str or Callable. If str, the add_to_pack
                function will call `setattr(entry, attribute_set, value)`.
                If Callable, the add_to_pack function will call
                `attribute_set(entry, value)`.

                If this key is not provided and the key "attribute_get" is
                a str type, "attribute_set" will be the same as "attribute_get".
                Otherwise, this key must be provided.
    """
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)

        assert hasattr(self.config, "attribute_get"), \
            "attribute_get is required in AttributeExtractor."

        self.attribute_get = self.config.attribute_get

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
        meta_data = {"pad_value": self.get_pad_value(),
                     "dim": 1,
                     "dtype": int if self.vocab else Any}
        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Iterable[Union[int, Any]]):
        """We make following assumptions for prediction.
        1. Prediction is not one-hot vector and we should
           use Vocabulary.id2element to map it back to the
           element we want to use for setting the attribute.
        2. If prediction is an interger, it means there is
           only one entry for an instance, and its attribute
           is specified by this prediction.
        3. If prediction is an iterable value (e.g. List, Array),
           we will truncate it according to the number of entry.
           If the prediction contains <PAD> element, this should remove them.
        """
        assert self.attribute_set != "text", "Text attribute is not"\
                                            "allowed to be set."
        instance_entry = list(pack.get(self.config.entry_type, instance))
<<<<<<< HEAD
        prediction = prediction[:len(instance_entry)]
        # TODO: we make some assumption here. The prediction is id.
=======

        if isinstance(prediction, abc.Iterable):
            prediction = prediction[:len(instance_entry)]
        else:
            assert len(instance_entry) == 1
            prediction = [prediction]
>>>>>>> 627331c... RJQ: [extractor] update design, TODO: link extractor
        attrs = [self.id2element(int(x)) for x in prediction]
        for entry, attr in zip(instance_entry, attrs):
            if callable(self.attribute_set):
                self.attribute_set(entry, attr)
            else:
                setattr(entry, self.attribute_set, attr)
