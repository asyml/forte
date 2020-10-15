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
Unit tests for data augment processors
"""

import unittest
import tempfile
import os

from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackSentenceReader
from forte.processors.base.data_augment_processor_tmp import ReplacementDataAugmentProcessor
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Token, Sentence, Document, Annotation

from ddt import ddt, data, unpack


@ddt
class TestReplacementDataAugmentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_replace_token(self, texts):
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

        nlp.initialize()

        expected_outputs = [
            "Virgin and Samantha arrived at the bus stop early but waited til 12 for the bus.\n"
        ]

        expected_tokens = [
            ["Virgin", "and", "Samantha", "arrived", "at", "the", "bus", "stop", "early", "but", "waited", "til", "12", "for", "the", "bus", "."],
        ]

        token_replacement = {
            "Mary": "Virgin",
            "noon": "12",
            "station": "stop",
            "until": "til",
        }

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'auto_align_entries': {
                "ft.onto.base_ontology.Sentence": "auto_align",
                "ft.onto.base_ontology.Document": "auto_align"
            }
        }

        processor = ReplacementDataAugmentProcessor()
        processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            data_pack = m_pack.get_pack("input_src")
            replaced_annotations = []
            for token in data_pack.get(Token):
                if token.text in token_replacement:
                    replaced_annotations.append((token, token_replacement[token.text]))
            new_pack = processor.replace_annotations(data_pack, replaced_annotations)

            self.assertEqual(new_pack.text, expected_outputs[idx])

            for j, token in enumerate(new_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

            for sent in new_pack.get(Sentence):
                self.assertEqual(sent.text, expected_outputs[idx].strip())


if __name__ == "__main__":
    unittest.main()
