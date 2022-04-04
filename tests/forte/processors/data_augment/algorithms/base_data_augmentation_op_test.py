# Copyright 2022 The Forte Authors. All Rights Reserved.
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
This module tests the working of the Base Op Processor
and the Base Augmentation Op along with its internal components.
"""

import os
import tempfile
import unittest

from typing import Union, Dict, Any
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence, Document, Annotation
from forte.data.span import Span
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import MultiPackLink, Link, MultiPackGroup, Group
from forte.utils.utils import get_class
from forte.data.readers import MultiPackSentenceReader, StringReader
from forte.data.caster import MultiPackBoxer
from forte.processors.misc import WhiteSpaceTokenizer
from forte.data.selector import AllPackSelector
from forte.processors.data_augment.base_op_processor import (
    BaseOpProcessor,
)
from forte.processors.data_augment.algorithms.base_data_augmentation_op import (
    BaseDataAugmentationOp,
)
from ddt import ddt, data, unpack


class DummyAugmenter(BaseDataAugmentationOp):
    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:
        super().__init__(configs)

    def augment(self, input_pack: DataPack) -> bool:
        try:
            # Collecting existing annotations. For this
            # example we use the Token class
            augment_entry = get_class("ft.onto.base_ontology.Token")
            annotation_list = []
            for anno in input_pack.get(augment_entry):
                annotation_list.append(anno)

            # Testing the insert annotations method
            new_text_1 = "There"
            new_text_2 = "Avoid"
            _ = self.insert_annotated_span(
                new_text_1, input_pack, 2, "ft.onto.base_ontology.Token"
            )
            _ = self.insert_annotated_span(
                new_text_2, input_pack, 5, "ft.onto.base_ontology.Sentence"
            )

            # Testing the delete annotations method
            _ = self.delete_annotation(annotation_list[2])
            _ = self.delete_span(input_pack, 7, 10)

            # Testing the replace annotations method
            replaced_text = "Last"
            _ = self.replace_annotations(annotation_list[-1], replaced_text)
            _ = self.replace_annotations(annotation_list[-1], replaced_text)
            return True
        except:
            return False


class ReplacementAugmentTest(BaseDataAugmentationOp):
    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:
        super().__init__(configs)

    def augment(self, input_pack: DataPack) -> bool:
        try:
            _ = _ = self.insert_span(" NLP ", input_pack, 0)
            _ = self.insert_span(" NLP ", input_pack, 18)
            _ = self.insert_span(" NLP ", input_pack, len(input_pack.text) - 2)
            _ = self.insert_span(" NLP", input_pack, len(input_pack.text) - 1)
            # Delete the second token "and"
            _ = self.delete_annotation(list(input_pack.get(Token))[1])

            # This duplicate insertion should be invalid.
            _ = self.insert_span(" NLP ", input_pack, 0)
            # This insertion overlaps with a replacement.
            # It should be invalid.
            _ = self.insert_span(" NLP ", input_pack, 2)

            return True

        except:
            return False


@ddt
class TestBaseOp(unittest.TestCase):
    def setUp(self) -> None:
        self.base_op = DummyAugmenter({})
        self.base_processor = BaseOpProcessor()
        self.test_dir = tempfile.mkdtemp()

    def test_operations(self) -> None:

        # Check if when initialized, is are the variables that hold
        # the inserted, deleted and replaced annotations are empty.
        insert_list = self.base_op._inserted_text
        delete_list = self.base_op._deleted_annos_id
        replace_list = self.base_op._replaced_annos
        self.assertFalse(insert_list)
        self.assertFalse(delete_list)
        self.assertFalse(replace_list)

        # Testing the insert annotations method
        data_pack = DataPack()
        data_pack.set_text("Hello World Today Welcome")
        token_1 = Token(data_pack, 0, 5)
        token_2 = Token(data_pack, 6, 11)
        token_3 = Token(data_pack, 12, 17)
        token_4 = Token(data_pack, 18, 25)
        token_5 = Sentence(data_pack, 12, 25)
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)
        data_pack.add_entry(token_3)
        data_pack.add_entry(token_4)
        data_pack.add_entry(token_5)

        # Perform predefined augmentation and check if
        # it worked correctly
        augmented_data_pack = self.base_op.perform_augmentation(data_pack)
        self.assertEqual(augmented_data_pack.text, "HeTherelloAvoid Wd  Last")

        insert_list = self.base_op._inserted_text
        delete_list = self.base_op._deleted_annos_id
        replace_list = self.base_op._replaced_annos

        self.assertEqual(
            list(insert_list[data_pack.pack_id].items()),
            [
                (2, (5, "ft.onto.base_ontology.Token")),
                (5, (5, "ft.onto.base_ontology.Sentence")),
            ],
        )

        self.assertEqual(list(delete_list[data_pack.pack_id]), [token_3.tid])

        expectations = [
            (Span(2, 2), "There"),
            (Span(5, 5), "Avoid"),
            (Span(7, 10), ""),
            (Span(12, 17), ""),
            (Span(18, 25), "Last"),
        ]

        self.assertEqual(list(replace_list[data_pack.pack_id]), expectations)

    @data(
        (1, [[0, 1], [1, 1], [1, 3]], [[0, 2], [2, 5], [5, 8]], True, True, 2),
        (1, [[0, 1], [1, 1], [2, 3]], [[0, 2], [2, 5], [6, 8]], True, True, 2),
        (1, [[0, 1], [1, 1], [1, 3]], [[0, 2], [2, 5], [5, 8]], True, False, 5),
        (1, [[0, 1], [1, 1], [2, 3]], [[0, 2], [2, 5], [6, 8]], True, False, 5),
        (1, [[0, 1], [1, 1], [1, 3]], [[0, 2], [2, 5], [5, 8]], False, True, 5),
        (1, [[0, 1], [1, 1], [2, 3]], [[0, 2], [2, 5], [6, 8]], False, True, 5),
        (
            1,
            [[0, 1], [1, 1], [1, 3]],
            [[0, 2], [2, 5], [5, 8]],
            False,
            False,
            2,
        ),
        (
            1,
            [[0, 1], [1, 1], [2, 3]],
            [[0, 2], [2, 5], [6, 8]],
            False,
            False,
            2,
        ),
        (0, [[1, 2], [2, 3]], [[1, 4], [4, 5]], True, True, 0),
    )
    @unpack
    def test_modify_index(
        self, index, old_spans, new_spans, is_begin, is_inclusive, aligned_index
    ):
        old_spans = [Span(span[0], span[1]) for span in old_spans]
        new_spans = [Span(span[0], span[1]) for span in new_spans]
        output = self.base_op.modify_index(
            index, old_spans, new_spans, is_begin, is_inclusive
        )
        self.assertEqual(aligned_index, output)

    def test_multi_pack_copy_link_or_group(self):
        m_pack = MultiPack()
        src_pack = m_pack.add_pack("src")
        tgt_pack = m_pack.add_pack("tgt")

        src_pack.set_text("input")
        tgt_pack.set_text("output")
        src_token = src_pack.add_entry(Token(src_pack, 0, len(src_pack.text)))
        tgt_token = tgt_pack.add_entry(Token(tgt_pack, 0, len(tgt_pack.text)))

        mpl = m_pack.add_entry(MultiPackLink(m_pack, src_token, tgt_token))
        # The MultiPackLink should not be copied, because its children are not copied.
        self.assertEqual(
            self.base_processor._copy_multi_pack_link_or_group(mpl, m_pack),
            False,
        )
        new_src_pack = self.base_op._apply_augmentations(src_pack)
        self.assertEqual(len(list(new_src_pack.get(Token))), 1)

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station early but waited \
                    until noon for the bus ."
            ],
            [
                "MaTherery Avoidand  arrived at the bus station early but waited \
                    until noon for the bus Last"
            ],
            [
                [
                    "MaTherery",
                    "There",
                    "and",
                    "arrived",
                    "at",
                    "the",
                    "bus",
                    "station",
                    "early",
                    "but",
                    "waited",
                    "until",
                    "noon",
                    "for",
                    "the",
                    "bus",
                    "Last",
                ]
            ],
        )
    )
    @unpack
    def test_pipeline(self, texts, expected_outputs, expected_tokens):
        nlp = Pipeline[MultiPack]()

        boxer_config = {"pack_name": "input"}

        replacer_op = (
            DummyAugmenter.__module__ + "." + DummyAugmenter.__qualname__
        )

        processor_config = {
            "data_aug_op": replacer_op,
            "data_aug_op_config": {
                "other_entry_policy": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align",
                },
            },
            "augment_pack_names": {},
        }

        nlp.set_reader(reader=StringReader())
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())
        nlp.add(component=BaseOpProcessor(), config=processor_config)
        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(texts)):
            aug_pack = m_pack.get_pack("augmented_input")

            self.assertEqual(aug_pack.text, expected_outputs[idx])

            for j, token in enumerate(aug_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station early but waited \
                    until noon for the bus ."
            ],
            [
                " NLP Ma NLP ry  Samantha  NLP arrived at the bus station early but waited \
                    until noon for the bus  NLP . NLP"
            ],
            [
                "Ma NLP ry  Samantha  NLP arrived at the bus station early but waited \
                    until noon for the bus  NLP ."
            ],
            [
                [
                    "Ma NLP ry",
                    "Samantha",
                    "arrived",
                    "at",
                    "the",
                    "bus",
                    "station",
                    "early",
                    "but",
                    "waited",
                    "until",
                    "noon",
                    "for",
                    "the",
                    "bus",
                    ".",
                ],
            ],
            [["til", "noon", "for", "the", "bus", "."]],
        )
    )
    @unpack
    def test_replace_token(
        self,
        texts,
        expected_outputs,
        expected_sentences,
        expected_tokens,
        expected_links,
    ):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, "w") as f:
                f.write(text)

        nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt",
        }
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)

        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())

        replacer_op = (
            ReplacementAugmentTest.__module__
            + "."
            + ReplacementAugmentTest.__qualname__
        )

        processor_config = {
            "data_aug_op": replacer_op,
            "data_aug_op_config": {
                "other_entry_policy": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Token": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align",
                },
            },
            "augment_pack_names": {},
        }

        nlp.initialize()

        processor = BaseOpProcessor()
        # To test, initialize the processor itself.
        processor.initialize(resources=None, configs=processor_config)

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            src_pack = m_pack.get_pack("input_src")
            tgt_pack = m_pack.get_pack("output_tgt")

            num_mpl_orig, num_mpg_orig = 0, 0
            # Copy the source pack to target pack.
            tgt_pack.set_text(src_pack.text)

            src_pack.add_entry(Document(src_pack, 0, len(src_pack.text)))
            for anno in src_pack.get(Annotation):
                new_anno = type(anno)(tgt_pack, anno.begin, anno.end)
                tgt_pack.add_entry(new_anno)

                # Create MultiPackLink.
                m_pack.add_entry(MultiPackLink(m_pack, anno, new_anno))

                # Create MultiPackGroup.
                m_pack.add_entry(MultiPackGroup(m_pack, [anno, new_anno]))

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
                    src_pack.add_entry(link)
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
                    tgt_pack.add_entry(group)
                    prev_entry = group
                else:
                    prev_entry = token

            processor._process(m_pack)

            new_src_pack = m_pack.get_pack("augmented_input_src")
            new_tgt_pack = m_pack.get_pack("augmented_output_tgt")

            self.assertEqual(new_src_pack.text, expected_outputs[idx] + "\n")

            for j, token in enumerate(new_src_pack.get(Token)):
                # print(f'[{token.text}], [{expected_tokens[idx][j]}]')
                self.assertEqual(token.text, expected_tokens[idx][j])

            for sent in new_src_pack.get(Sentence):
                self.assertEqual(sent.text, expected_sentences[idx])

            # Test the copied Links.
            prev_link = None
            for i, link in enumerate(new_src_pack.get(Link)):
                if prev_link:
                    self.assertEqual(link.get_parent().tid, prev_link.tid)
                    self.assertEqual(
                        link.get_child().text, expected_links[idx][i]
                    )
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
                    self.assertEqual(member_token.text, expected_links[idx][i])

                prev_group = group

            # The two extra Links should not be copied, because of missing Document.
            self.assertEqual(
                len(list(src_pack.get(Link))),
                len(list(new_src_pack.get(Link))),
            )
            # The two extra Groups should not be copied, because of missing Document.
            self.assertEqual(
                len(list(tgt_pack.get(Group))),
                len(list(new_tgt_pack.get(Group))),
            )

            # Test the MultiPackLink/MultiPackGroup
            num_mpl_aug, num_mpg_aug = 0, 0
            for mpl in m_pack.get(MultiPackLink):
                parent = mpl.get_parent()
                child = mpl.get_child()
                num_mpl_aug += 1
                self.assertEqual(parent.text, child.text)
                self.assertNotEqual(parent.pack.pack_id, child.pack.pack_id)

            for mpg in m_pack.get(MultiPackGroup):
                members = mpg.get_members()
                num_mpg_aug += 1
                self.assertEqual(members[0].text, members[1].text)
                self.assertNotEqual(
                    members[0].pack.pack_id, members[1].pack.pack_id
                )

            # Test the number of MultiPackLink/MultiPackGroup.
            # Minus the aug and orig counters by 1, because the Document is
            # not copied.
            # So we ignore the MPL and MPG between Document.
            # The number should be doubled, except for one deletion.
            self.assertEqual(num_mpl_aug, (num_mpl_orig) * 2 - 1)
            self.assertEqual(num_mpg_aug, (num_mpg_orig) * 2 - 1)


if __name__ == "__main__":
    unittest.main()
