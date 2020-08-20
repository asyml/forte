# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit tests for Clinical NER processors.
"""

import unittest
import os

from examples.Cliner.Cliner import ClinicalNER
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from ft.onto.clinical import ClinicalEntityMention


class TestClinicalNER(unittest.TestCase):
    def setUp(self):
        self.cliner = Pipeline[DataPack]()
        self.cliner.set_reader(StringReader())
        self.cliner.add(
            ClinicalNER(),
            config={
                'config_model':
                'examples/Cliner/CliNER/models/train_full'
                '.model',
                'config_data':
                'examples/Cliner/CliNER/data/examples/test.txt',
                'config_output':
                'examples/Cliner/CliNER/data/examples',
            })
        self.cliner.initialize()

    def test_ner(self):
        sentences = [
            "Her fall was not observed , but the patient does not "
            "profess any loss of consciousness , recalling the "
            "entire event . ",
            "The patient does have a history of previous falls , "
            "one of which resulted in a hip fracture . ",
            "Initial examination showed bruising around the left eye "
            ", normal lung examination , normal heart examination , "
            "normal neurologic function with a baseline decreased "
            "mobility of her left arm . ",
        ]
        document = '\n'.join(sentences)
        with open('examples/Cliner/CliNER/data/examples/test.txt',
                  'w') as fo:
            fo.write(document)
        pack = self.cliner.process(document)

        entities_entries = list(pack.get(entry_type=ClinicalEntityMention))

        entities_text = [x.text for x in entities_entries]
        entities_type = [x.ner_type for x in entities_entries]

        self.assertEqual(entities_text, [
            'loss of consciousness', 'previous falls', 'a hip fracture',
            'Initial examination'
        ])
        self.assertEqual(entities_type,
                         ['problem', 'problem', 'problem', 'test'])


if __name__ == "__main__":
    unittest.main()
