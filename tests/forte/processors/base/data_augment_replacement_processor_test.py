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

from forte.processors.data_augment.algorithms.text_replacement_op import TextReplacementOp
from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import MultiPackLink
from forte.data.readers import MultiPackSentenceReader
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Token, Sentence, Document, Annotation

from ddt import ddt, data, unpack

__all__ = [
    "Replacer"
]


class TmpReplacementDataAugmentProcessor(ReplacementDataAugmentProcessor):
    def new_pack(self):
        return MultiPack()


class Replacer(TextReplacementOp):
    def __init__(self, configs):
        super().__init__(configs)
        self.token_replacement = {
            "Mary": "Virgin",
            "noon": "12",
            "station": "stop",
            "until": "til",
        }

    def replace(self, input):
        return self.token_replacement.get(input.text, input.text)


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

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "ft.onto.base_ontology.Sentence": "auto_align",
                "ft.onto.base_ontology.Document": "auto_align"
            },
            'replacement_op': "tests.forte.processors.base.data_augment_replacement_processor_test.Replacer",
            'replacement_op_config': {}
        }

        processor = TmpReplacementDataAugmentProcessor()
        processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
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

            processor._process(m_pack)

            new_src_pack = m_pack.get_pack('augmented_input_src')

            self.assertEqual(new_src_pack.text, expected_outputs[idx])

            for j, token in enumerate(new_src_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

            for sent in new_src_pack.get(Sentence):
                self.assertEqual(sent.text, expected_outputs[idx].strip())

            for mpl in m_pack.get(MultiPackLink):
                parent = mpl.get_parent()
                child = mpl.get_child()
                self.assertEqual(parent.text, child.text)
                self.assertNotEqual(parent.pack.meta.pack_id, child.pack.meta.pack_id)


if __name__ == "__main__":
    unittest.main()
