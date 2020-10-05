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

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackSentenceReader
from forte.processors.data_augment.text_generation_augment_processor import TextGenerationDataAugmentProcessor
from forte.processors.data_augment.algorithms.dictionary_replacement_augmenter import DictionaryReplacementAugmenter
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Sentence

from ddt import ddt, data, unpack

# Manually test


from ft.onto.base_ontology import Token, Sentence
data_pack = DataPack()
data_pack.set_text("Mary and Samantha. I love NLP.")
data_pack.add_entry(Token(data_pack, 0, 4))
data_pack.add_entry(Token(data_pack, 5, 8))
data_pack.add_entry(Token(data_pack, 9, 17))
data_pack.add_entry(Token(data_pack, 17, 18))
data_pack.add_entry(Token(data_pack, 19, 20))
data_pack.add_entry(Token(data_pack, 21, 25))
data_pack.add_entry(Token(data_pack, 26, 29))
data_pack.add_entry(Token(data_pack, 29, 30))
data_pack.add_entry(Sentence(data_pack, 0, 18))
data_pack.add_entry(Sentence(data_pack, 18, 30))



class Config:
    def __init__(self):
        self.augment_entries = ["Token", "Sentence", "Document"]
        self.replacement_prob =  0.9
        self.replacement_level = 'word'
        self.input_pack_name = 'input_src'
        self.output_pack_name =  'output_tgt'
        self.aug_input_pack_name = 'aug_input_src'
        self.aug_output_pack_name = 'aug_output_tgt'
        self.aug_num = 1

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

data_augment_config = Config()

augmenter = DictionaryReplacementAugmenter({"lang": "eng"})

processor = TextGenerationDataAugmentProcessor()
processor.initialize(resources=None, configs=data_augment_config)
processor.augmenter = augmenter
new_pack = processor._process_pack(data_pack)
new_pack.add_all_remaining_entries()
print(new_pack.text)
for token in new_pack.get(Token):
    print(token.text)
for sent in new_pack.get(Sentence):
    print(sent.text)

exit()


### Manual test end.

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
        nlp.add(NLTKWordTokenizer())
        nlp.add(NLTKPOSTagger())

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
