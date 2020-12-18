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


from typing import List, Dict, Union
from ft.onto.base_ontology import Annotation, EntityMention
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.utils import bio_tagging
from forte.data.extractor.base_extractor import BaseExtractor


class BioSeqTaggingExtractor(BaseExtractor):
    """BioSeqTaggingExtractor will the feature by performing BIO encoding
    for the attribute of entry, aligining to the based_on entry.
    Args:
        config:
            Required keys:
            "attribute": str. The attribute of entry needed to be encoded.
            "based_on": Type[Annotation]. The entry that the attribute should
                align to.
    """
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)

        assert hasattr(self.config, "attribute"), \
            "attribute is required in BioSeqTaggingExtractor."
        assert hasattr(self.config, "based_on"), \
            "based_on is required in BioSeqTaggingExtractor."

    def bio_variance(self, tag):
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def predefined_vocab(self, predefined: set):
        for tag in predefined:
            for element in self.bio_variance(tag):
                self.add(element)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            for tag_variance in self.bio_variance(attribute):
                self.add(tag_variance)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        instance_based_on = list(pack.get(self.config.based_on, instance))
        instance_entry = list(pack.get(self.config.entry_type, instance))
        instance_tagged = bio_tagging(instance_based_on, instance_entry)

        data = []
        for pair in instance_tagged:
            if pair[0] is None:
                new_pair = (None, pair[1])
            else:
                new_pair = (getattr(pair[0], self.config.attribute), pair[1])
            if self.vocab:
                data.append(self.element2repr(new_pair))
            else:
                data.append(new_pair)
        meta_data = {"pad_value": self.get_pad_value(),
                     "dim": 1,
                     "dtype": int if self.vocab else tuple}
        return Feature(data=data,
                       metadata=meta_data,
                       vocab=self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: List[int]):
        """We make following assumptions for prediction.
        1. If we encounter "I" while its tag is different from the previous tag,
           we will consider this "I" as a "B" and start a new tag here.
        2. We will truncate the prediction it according to the number of entry.
           If the prediction contains <PAD> element, this should remove them.
        3. We will remove the entry before we add new entry from the prediction.
        """
        for entry in pack.get(self.config.entry_type, instance):
            pack.delete_entry(entry)
        instance_based_on = list(pack.get(self.config.based_on, instance))
        prediction = prediction[:len(instance_based_on)]
        tags = [self.id2element(x) for x in prediction]
        tag_start = None
        tag_end = None
        tag_type = None
        cnt = 0
        for entry, tag in zip(instance_based_on, tags):
            if tag[1] == "O" or tag[1] == "B":
                if tag_type:
                    entity_mention = EntityMention(pack, tag_start, tag_end)
                    entity_mention.ner_type = tag_type
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            else:
                if tag[0] == tag_type:
                    tag_end = entry.end
                else:
                    if tag_type:
                        entity_mention = EntityMention(pack, tag_start, tag_end)
                        entity_mention.ner_type = tag_type
                    tag_start = entry.begin
                    tag_end = entry.end
                    tag_type = tag[0]
            cnt += 1

        # Handle the final tag
        if tag_type:
            entity_mention = EntityMention(pack, tag_start, tag_end)
            entity_mention.ner_type = tag_type
