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
import random

from forte.processors.data_augment.algorithms.text_replacement_op import TextReplacementOp
from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import MultiPackLink
from forte.data.readers import MultiPackSentenceReader
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from forte.processors.data_augment.algorithms.eda_processors \
    import RandomDeletionDataAugmentProcessor, RandomInsertionDataAugmentProcessor, \
    RandomSwapDataAugmentProcessor
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Token, Sentence, Document, Annotation

from ddt import ddt, data, unpack


class TmpReplacementDataAugmentProcessor(ReplacementDataAugmentProcessor):
    def new_pack(self):
        return MultiPack()


class TmpReplacer(TextReplacementOp):
    def __init__(self, configs={}):
        super().__init__(configs)
        self.token_replacement = {
            "Mary": "Virgin",
            "noon": "12",
            "station": "stop",
            "until": "til",
        }

    def replace(self, input):
        return True, self.token_replacement.get(input.text, input.text)


@ddt
class TestEDADataAugmentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        random.seed(0)
        self.nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }
        self.nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)

        self.nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
        self.nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())

        self.nlp.initialize()

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_random_swap(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)


        expected_text = "Mary early Samantha arrived at the bus station and but waited until for noon the .bus\n"
        expected_tokens = ['Mary', 'early', 'Samantha', 'arrived', 'at', 'the', 'bus', 'station', 'and', 'but', 'waited', 'until', 'for', 'noon', 'the', '.', 'bus']

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            "kwargs": {},
            'n': 3,
        }

        swap_processor = RandomSwapDataAugmentProcessor()
        swap_processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(self.nlp.process_dataset(self.test_dir)):
            src_pack = m_pack.get_pack('input_src')
            tgt_pack = m_pack.get_pack('output_tgt')

            # Copy the source pack to target pack.
            tgt_pack.set_text(src_pack.text)
            for anno in src_pack.get(Annotation):
                new_anno = type(anno)(
                    tgt_pack, anno.begin, anno.end
                )
                tgt_pack.add_entry(new_anno)

                m_pack.add_entry(
                    MultiPackLink(
                        m_pack, anno, new_anno
                    )
                )

            swap_processor._process(m_pack)

            new_src_pack = m_pack.get_pack('augmented_input_src')

            print(new_src_pack.text)
            self.assertEqual(new_src_pack.text, expected_text)
            for j, token in enumerate(new_src_pack.get(Token)):
                print(token.text, expected_tokens[j])
                self.assertEqual(token.text, expected_tokens[j])


    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_random_insert(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        expected_tokens = ['await ', 'await Mary', ' Mary and', 'y and Samantha', 'antha arrived', 'rived at', 'ed at the', 't the bus', 'e bus station', 'ation early', ' station', 'station but', 'ion but waited', ' waited until', 'd until noon', 'il noon for', 'oon for the', 'for the bus', ' station', ' station.']

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            "kwargs": {
                'data_aug_op': "forte.processors.data_augment.algorithms.dictionary_replacement_op.DictionaryReplacementOp",
                'data_aug_op_config': {
                    "dictionary": (
                        "forte.processors.data_augment."
                        "algorithms.dictionary.WordnetDictionary"
                    ),
                    "prob": 1.0,
                    "lang": "eng",
                }
            },
            'n': 1,
        }

        insert_processor = RandomInsertionDataAugmentProcessor()
        insert_processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(self.nlp.process_dataset(self.test_dir)):
            src_pack = m_pack.get_pack('input_src')
            tgt_pack = m_pack.get_pack('output_tgt')

            # Copy the source pack to target pack.
            tgt_pack.set_text(src_pack.text)
            for anno in src_pack.get(Annotation):
                new_anno = type(anno)(
                    tgt_pack, anno.begin, anno.end
                )
                tgt_pack.add_entry(new_anno)

                m_pack.add_entry(
                    MultiPackLink(
                        m_pack, anno, new_anno
                    )
                )

            insert_processor._process(m_pack)

            new_src_pack = m_pack.get_pack('augmented_input_src')

            print(new_src_pack.text)
            result = []
            for j, token in enumerate(new_src_pack.get(Token)):
                result.append(token.text)
            print(result)

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_random_delete(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            "prob": 0.5,
        }

        delete_processor = RandomDeletionDataAugmentProcessor()
        delete_processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(self.nlp.process_dataset(self.test_dir)):
            src_pack = m_pack.get_pack('input_src')
            tgt_pack = m_pack.get_pack('output_tgt')

            # Copy the source pack to target pack.
            tgt_pack.set_text(src_pack.text)
            for anno in src_pack.get(Annotation):
                new_anno = type(anno)(
                    tgt_pack, anno.begin, anno.end
                )
                tgt_pack.add_entry(new_anno)

                m_pack.add_entry(
                    MultiPackLink(
                        m_pack, anno, new_anno
                    )
                )

            delete_processor._process(m_pack)

            new_src_pack = m_pack.get_pack('augmented_input_src')

            print(new_src_pack.text)
            result = []
            for j, token in enumerate(new_src_pack.get(Token)):
                result.append(token.text)
            print(result)


if __name__ == "__main__":
    unittest.main()
