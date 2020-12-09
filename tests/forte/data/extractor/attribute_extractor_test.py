#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest

from ft.onto.base_ontology import Sentence, Token, EntityMention
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.extractor.attribute_extractor import AttributeExtractor


class AttributeExtractorTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03_new"

    def test_AttributeExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config1 = {
            "scope": Sentence,
            "entry_type": Token,
            "attribute_get": "text",
        }

        config2 = {
            "scope": Sentence,
            "entry_type": Token,
            "attribute_get": "text",
            # This must be a field that could be set.
            "attribute_set": "chunk"
        }

        config3 = {
            "scope": Sentence,
            "entry_type": Token,
            "attribute_get": lambda x: x.text,
            "attribute_set": "chunk"
        }

        config4 = {
            "scope": Sentence,
            "entry_type": Token,
            "attribute_get": lambda x: x.text,
            "attribute_set": lambda x, value:
                setattr(x, "chunk", value)
        }

        for config in [config1, config2, config3, config4]:
            extractor = AttributeExtractor(config)

            sentence = "EU rejects German call to boycott British lamb ."

            for pack in pipeline.process_dataset(self.dataset_path):
                for instance in pack.get(Sentence):
                    extractor.update_vocab(pack, instance)

            for pack in pipeline.process_dataset(self.dataset_path):
                features = []
                for instance in pack.get(Sentence):
                    features.append(extractor.extract(pack, instance))

                for feat in features:
                    recovered = [extractor.id2element(idx) for idx in feat._data]
                    self.assertEqual(" ".join(recovered), sentence)
                    if extractor.attribute_set != "text":
                        extractor.add_to_pack(pack, instance, recovered)


if __name__ == '__main__':
    unittest.main()
