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

from forte.data.batchers import FixedSizeDataPackBatcherWithExtractor
from forte.data.converter import Converter
from forte.data.data_pack import DataPack
from forte.data.extractors.attribute_extractor import AttributeExtractor
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.pipeline import Pipeline
from forte.train_preprocessor import TrainPreprocessor
from ft.onto.base_ontology import Sentence, Token


class FixedSizeDataPackBatcherWithExtractorTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03"

    def test_FixedSizeDataPackBatcherWithExtractor(self):
        r"""This funciton tests the corectness of cross_pack."""
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(CoNLL03Reader())
        pipeline.initialize()

        text_extractor = AttributeExtractor({
            "need_pad": True,
            "entry_type": Token,
            "attribute": "text",
        })

        pack_num = 0
        for pack in pipeline.process_dataset(self.dataset_path):
            pack_num += 1
            for instance in pack.get(Sentence):
                text_extractor.update_vocab(pack, instance)
        self.assertEqual(pack_num, 2)

        batch_size = 2
        batcher = FixedSizeDataPackBatcherWithExtractor(cross_pack=True)
        batcher.initialize({
            "scope": Sentence,
            "batch_size": batch_size,
            "feature_scheme": {
                "text_tag": {
                    "extractor": text_extractor,
                    "converter": Converter(),
                    "type": TrainPreprocessor.DATA_INPUT
                }
            },
        })

        batch_num = 0
        for pack in pipeline.process_dataset(self.dataset_path):
            for batch in batcher.get_batch(pack, Sentence, None):
                batch_num += 1
                self.assertEqual(len(batch[0]), batch_size)
        for _ in batcher.flush():
            batch_num += 1
        self.assertEqual(batch_num, 1)


if __name__ == '__main__':
    unittest.main()
