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
"""
Unit tests for text generation augment processors
"""

import unittest
import tempfile
import os

from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackSentenceReader
from forte.processors.data_augment.text_generation_augment_processor import TextGenerationDataAugmentProcessor
from forte.processors.data_augment.algorithms.dictionary_replacement_augmenter import DictionaryReplacementAugmenter
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger

from ddt import ddt, data, unpack


@ddt
class TestTextGenerationAugmentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)

        nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
        nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())

        data_augment_config = {
            'augment_entries': ["Token", "Sentence", "Document"],
            'replacement_prob': 0.9,
            'replacement_level': 'word',
            'input_pack_name': 'input_src',
            'output_pack_name': 'output_tgt',
            'aug_input_pack_name': 'aug_input_src',
            'aug_output_pack_name': 'aug_output_tgt',
            'aug_num': 1,
        }

        augmenter = DictionaryReplacementAugmenter({"lang": "eng"})

        processor = TextGenerationDataAugmentProcessor()
        processor.augmenter = augmenter

        nlp.add(processor,
                config=data_augment_config)
        nlp.initialize()

        expected_outputs = [
            "Blessed Virgin and Samantha go far at the motorbus station early on but wait until twelve noon for the bus topology."
        ]

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            self.assertEqual(m_pack.get_pack("aug_input_src_0").text, expected_outputs[idx])


if __name__ == "__main__":
    unittest.main()
