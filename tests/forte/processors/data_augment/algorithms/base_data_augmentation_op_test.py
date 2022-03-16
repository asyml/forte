import unittest

from typing import Union, Dict, Any
from ft.onto.base_ontology import Token, Sentence
from forte.data.span import Span
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import MultiPackLink
from forte.utils.utils import get_class
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
            insert_attempt_1 = self.insert_annotated_span(
                new_text_1, input_pack, 2, "ft.onto.base_ontology.Token"
            )
            insert_attempt_2 = self.insert_annotated_span(
                new_text_2, input_pack, 5, "ft.onto.base_ontology.Sentence"
            )

            # Testing the delete annotations method
            delete_attempt_1 = self.delete_annotation(annotation_list[2])
            delete_attempt_2 = self.delete_span(input_pack, 7, 10)

            # Testing the replace annotations method
            replaced_text = "Last"
            replace_attempt_1 = self.replace_annotations(
                annotation_list[-1], replaced_text
            )
            replace_attempt_2 = self.replace_annotations(
                annotation_list[-1], replaced_text
            )
            return True
        except:
            return False


@ddt
class TestBaseOp(unittest.TestCase):
    def setUp(self) -> None:
        self.base_op = DummyAugmenter({})

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
            self.base_op._copy_multi_pack_link_or_group(mpl, m_pack), False
        )
        new_src_pack = self.base_op._apply_augmentations(src_pack)
        self.assertEqual(len(list(new_src_pack.get(Token))), 1)


if __name__ == "__main__":
    unittest.main()
