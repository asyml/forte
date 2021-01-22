#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest
from ft.onto.base_ontology import Sentence, Token
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.converter import Converter
from forte.data.extractor.attribute_extractor import AttributeExtractor
from forte.data.extractor_batchers import FixedSizeDataPackBatcher


class ProcessingBatcher(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03"

    def test_AttributeExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "need_pad": True,
            "entry_type": Token,
            "attribute": "text",
        }

        extractor = AttributeExtractor(config)
        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        converter = Converter({})

        batcher = FixedSizeDataPackBatcher()

        batcher.initialize({
            "scope": Sentence,
            "feature_scheme": {
                "text_tag": {
                    "extractor": extractor,
                    "converter": converter
                }
            },
            "batch_size": 2
        })

        expected_packs = []
        batches = []

        for pack in pipeline.process_dataset(self.dataset_path):
            expected_packs.append(pack)
            for batch in batcher.get_batch(pack):
                batches.append(batch)
        for batch in batcher.flush():
            batches.append(batch)


        expected_instances = list(expected_packs[0].get(Sentence))+\
                        list(expected_packs[1].get(Sentence))
        sentence = "The European Commission said on Thursday it disagreed "\
                    "with German advice to consumers to shun British lamb "\
                    "until scientists determine whether mad cow disease "\
                    "can be transmitted to sheep ."

        self.assertEqual(len(batches), 1)

        packs, instances, features = batches[0]

        self.assertListEqual(expected_packs, packs)
        self.assertListEqual(expected_instances, instances)
        self.assertEqual(features['text_tag']['data'].shape[0], 2)
        recovered0 = [extractor.id2element(idx) for idx in
                    features['text_tag']['data'].numpy()[0]]
        recovered1 = [extractor.id2element(idx) for idx in
                    features['text_tag']['data'].numpy()[1]]
        self.assertEqual(" ".join(recovered0), sentence)
        self.assertEqual(" ".join(recovered1), sentence)


if __name__ == '__main__':
    unittest.main()
