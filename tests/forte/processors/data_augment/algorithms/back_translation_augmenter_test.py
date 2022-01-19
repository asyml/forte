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
Unit tests for back translation replacement op.
"""

import unittest
import random
from ft.onto.base_ontology import Sentence
from forte.data.data_pack import DataPack
from forte.processors.data_augment.algorithms.back_translation_op import (
    BackTranslationOp,
)


class TestBackTranslationAugmenter(unittest.TestCase):
    def setUp(self):
        model_class_name = (
            "forte.processors.data_augment.algorithms."
            "machine_translator.MarianMachineTranslator"
        )
        self.bta = BackTranslationOp(
            configs={
                "prob": 1.0,
                "model_to": model_class_name,
                "model_back": model_class_name,
                "src_language": "en",
                "tgt_language": "fr",
                "device": "cpu",
            }
        )

    def test_back_translation(self):
        random.seed(0)
        data_pack = DataPack()
        text = "Natural Language Processing has never been made this simple!"
        data_pack.set_text(text)
        sent = Sentence(data_pack, 0, len(text))
        data_pack.add_entry(sent)

        translated_text = (
            "The treatment of natural language has never been easier!"
        )
        assert translated_text == self.bta.replace(sent)[1]


if __name__ == "__main__":
    unittest.main()
