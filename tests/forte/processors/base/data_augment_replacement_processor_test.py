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
from forte.data.ontology.top import MultiPackLink, MultiPackGroup, Link, Group
from forte.data.readers import MultiPackSentenceReader
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Token, Sentence, Document, Annotation
# from forte.processors.stanfordnlp_processor import StandfordNLPProcessor
# from forte.common.configuration import Config

from ddt import ddt, data, unpack


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
class TestReplacementDataAugmentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    @data((["Mary and Samantha arrived at the bus station early but waited until noon for the bus."],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        expected_outputs = [
            "Virgin and Samantha arrived at the bus stop early but waited til 12 for the bus.\n"
        ]

        expected_tokens = [
            ["Virgin", "and", "Samantha", "arrived", "at", "the", "bus", "stop", "early", "but", "waited", "til", "12", "for", "the", "bus", "."],
        ]

        nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                'type': '',
                'kwargs': {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            'type': 'data_augmentation_op',
            'data_aug_op': 'tests.forte.processors.base.data_augment_replacement_processor_test.TmpReplacer',
            'data_aug_op_config': {
                'type': '',
                'kwargs': {}
            }
        }

        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
        nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())
        nlp.add(component=ReplacementDataAugmentProcessor(), config=processor_config)
        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            new_src_pack = m_pack.get_pack('augmented_input_src')

            self.assertEqual(new_src_pack.text, expected_outputs[idx])

            for j, token in enumerate(new_src_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

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

        # config = Config({
        #     'processors': 'tokenize,pos,lemma,depparse',
        #     'lang': "en",
        #     'use_gpu': False,
        # }, StandfordNLPProcessor.default_configs())
        # nlp.add(component=StandfordNLPProcessor(), config=config, selector=AllPackSelector())

        nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
        nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())

        nlp.initialize()

        expected_outputs = [
            " NLP Virgin  Samantha  NLP arrived at the bus stop early but waited til 12 for the bus NLP .\n"
        ]

        expected_tokens = [
            [" NLP ", "Virgin", "Samantha", " NLP ", "arrived", "at", "the", "bus", "stop", "early", "but", "waited", "til", "12", "for", "the", "bus", " NLP ", "."],
        ]

        expected_links = [
            "til", "12", "for", "the", "bus", "."
        ]

        processor_config = {
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "kwargs": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            'type': 'data_augmentation_op',
            'data_aug_op': 'tests.forte.processors.base.data_augment_replacement_processor_test.TmpReplacer',
            "data_aug_op_config": {
                'kwargs': {}
            }
        }

        processor = ReplacementDataAugmentProcessor()
        processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            src_pack = m_pack.get_pack('input_src')
            tgt_pack = m_pack.get_pack('output_tgt')

            num_mpl_orig, num_mpg_orig = 0, 0
            # Copy the source pack to target pack.
            tgt_pack.set_text(src_pack.text)
            for anno in src_pack.get(Annotation):
                new_anno = type(anno)(
                    tgt_pack, anno.begin, anno.end
                )
                tgt_pack.add_entry(new_anno)

                # Create MultiPackLink.
                m_pack.add_entry(
                    MultiPackLink(
                        m_pack, anno, new_anno
                    )
                )

                # Create MultiPackGroup.
                m_pack.add_entry(
                    MultiPackGroup(
                        m_pack, [anno, new_anno]
                    )
                )

                # Count the number of MultiPackLink/MultiPackGroup.
                num_mpl_orig += 1
                num_mpg_orig += 1

            # Create Links in the source pack.
            # The Links should be a tree:
            #
            #                           Link 3
            #                    _________|_________
            #                   |                  |
            #                 Link 2               |
            #            _______|________          |
            #           |               |          |
            #         Link 1            |          |
            #     ______|_____          |          |
            #    |           |          |          |
            # token 1     token 2    token 3    token 4 ... ...
            prev_entry = None
            for i, token in enumerate(src_pack.get(Token)):
                # Avoid overlapping with deleted tokens.
                if i < 10:
                    continue
                if prev_entry:
                    link = Link(src_pack, prev_entry, token)
                    src_pack.add_entry(
                        link
                    )
                    prev_entry = link
                else:
                    prev_entry = token

            # Create Groups in the target pack.
            # The Groups should be a tree like the Links.
            prev_entry = None
            for i, token in enumerate(tgt_pack.get(Token)):
                # Avoid overlapping with deleted tokens.
                if i < 10:
                    continue
                if prev_entry:
                    group = Group(tgt_pack, [prev_entry, token])
                    tgt_pack.add_entry(
                        group
                    )
                    prev_entry = group
                else:
                    prev_entry = token

            # Test the insertion and deletion
            for pack in (src_pack, tgt_pack):
                # Insert an "NLP" at the beginning
                processor._insert(" NLP ", pack, 0)
                processor._insert(" NLP ", pack, 18)
                processor._insert(" NLP ", pack, len(pack.text) - 2)
                # Delete the second token "and"
                processor._delete(list(pack.get(Token))[1])

            processor._process(m_pack)

            new_src_pack = m_pack.get_pack('augmented_input_src')
            new_tgt_pack = m_pack.get_pack('augmented_output_tgt')

            self.assertEqual(new_src_pack.text, expected_outputs[idx])

            for j, token in enumerate(new_src_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

            for sent in new_src_pack.get(Sentence):
                self.assertEqual(sent.text.strip(), expected_outputs[idx].strip())

            # Test the copied Links.
            prev_link = None
            for i, link in enumerate(new_src_pack.get(Link)):
                if prev_link:
                    self.assertEqual(link.get_parent().tid, prev_link.tid)
                    self.assertEqual(link.get_child().text, expected_links[i])
                prev_link = link

            # Test the copied Groups.
            prev_group = None
            for i, group in enumerate(new_tgt_pack.get(Group)):
                members = group.get_members()
                if isinstance(members[0], Token):
                    member_token = members[0]
                    member_group = members[1]
                else:
                    member_token = members[1]
                    member_group = members[0]

                if prev_group:
                    self.assertEqual(isinstance(member_token, Token), True)
                    self.assertEqual(isinstance(member_group, Group), True)
                    self.assertEqual(member_group.tid, prev_group.tid)
                    self.assertEqual(member_token.text, expected_links[i])

                prev_group = group

            # Test the MultiPackLink/MultiPackGroup
            num_mpl_aug, num_mpg_aug = 0, 0
            for mpl in m_pack.get(MultiPackLink):
                parent = mpl.get_parent()
                child = mpl.get_child()
                num_mpl_aug += 1
                self.assertEqual(parent.text, child.text)
                self.assertNotEqual(parent.pack.meta.pack_id, child.pack.meta.pack_id)

            for mpg in m_pack.get(MultiPackGroup):
                members = mpg.get_members()
                num_mpg_aug += 1
                self.assertEqual(members[0].text, members[1].text)
                self.assertNotEqual(members[0].pack.meta.pack_id, members[1].pack.meta.pack_id)

            # Test the number of MultiPackLink/MultiPackGroup.
            # The number should be doubled, except for one deletion.
            self.assertEqual(num_mpl_aug, num_mpl_orig * 2 - 1)
            self.assertEqual(num_mpg_aug, num_mpg_orig * 2 - 1)


if __name__ == "__main__":
    unittest.main()
