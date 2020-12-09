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
from ft.onto.base_ontology import Annotation, EntityMention
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.utils import bio_tagging
from forte.data.extractor.base_extractor import BaseExtractor


class BioSeqTaggingExtractor(BaseExtractor):
    '''This class will create a BIO tagging sequence for the attribute of
    the entry_type. E.g. the ner_type of EntityMention in a Sentence. The
    extracted sequence length will be the same as the based_on Annotation.
    E.g. the Token text of a Sentence.
    '''
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
            "based_on": None
        }
        self.config = Config(self.config,
                                default_hparams=defaults,
                                allow_new_hparam=True)
        assert self.config.attribute is not None, \
            "Attribute should not be None."
        assert self.config.based_on is not None, \
            "Based_on should not be None."

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
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 1,
                    "dtype": int if self.vocab else tuple}
        return Feature(data=data, metadata=meta_data,
                        vocab=self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        '''This function add the output tag back to the pack. If we
        encounter "I" while its tag is different from the previous tag,
        we will consider this "I" as a "B" and start a new tag here.
        '''
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
