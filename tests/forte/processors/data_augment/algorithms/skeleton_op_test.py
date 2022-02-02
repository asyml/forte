import unittest

from forte.data.span import Span
from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack
from typing import Union, Dict, Any
from forte.common.configuration import Config
from forte.utils.utils import get_class
from forte.processors.data_augment.algorithms.skeleton_op import (
    SkeletonOp
)

class DummyAugmenter(SkeletonOp):
    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:
        super().__init__(configs)

    def augment(self, input_pack: DataPack) -> bool:
        try:
            # Collecting existing annotations
            augment_entry = get_class(self.configs["augment_entry"])
            annotation_list = []
            for anno in input_pack.get(augment_entry):
                annotation_list.append(anno)

            # Testing the insert annotations method
            new_text_1 = "There"
            new_text_2 = "Avoid"
            insert_attempt_1 = self.insert_annotations(new_text_1, input_pack, 5)
            insert_attempt_2 = self.insert_annotations(new_text_2, input_pack, 5)

            # Testing the delete annotations method
            delete_attempt_1 = self.delete_annotations(annotation_list[0])
            delete_attempt_2 = self.delete_annotations(annotation_list[0])

            # Testing the replace annotations method
            replaced_text = "Last"
            replace_attempt_1 = self.replace_annotations(annotation_list[-1], replaced_text)
            replace_attempt_2 = self.replace_annotations(annotation_list[-1], replaced_text)
            return True
        except:
            return False

class TestSkeletonOp(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton_op = DummyAugmenter({})

    def test_operations(self) -> None:

        # Check if when initialized, is are the variables that hold
        # the inserted, deleted and replaced annotations are empty.
        insert_list = self.skeleton_op.inserted_annotation_status()
        delete_list = self.skeleton_op.deleted_annotation_status()
        replace_list = self.skeleton_op.replaced_annotation_status()
        self.assertFalse(insert_list)
        self.assertFalse(delete_list)
        self.assertFalse(replace_list)


        # Testing the insert annotations method
        data_pack = DataPack()
        data_pack.set_text("Hello World Today")
        token_1 = Token(data_pack, 0, 5)
        token_2 = Token(data_pack, 6, 11)
        token_3 = Token(data_pack, 12, 17)
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)
        data_pack.add_entry(token_3)

        # Perform predefined augmentation and check if
        # it worked correctly
        success = self.skeleton_op.augment(data_pack)
        self.assertTrue(success)

        insert_list = self.skeleton_op.inserted_annotation_status()
        delete_list = self.skeleton_op.deleted_annotation_status()
        replace_list = self.skeleton_op.replaced_annotation_status()

        self.assertEqual(
            list(insert_list[data_pack.pack_id].items()),
            [(5, 5)]
        )

        self.assertEqual(
            list(delete_list[data_pack.pack_id]),
            [token_1.tid]
        )

        expectations = [
                (Span(token_1.begin, token_1.end), ""),
                (Span(token_1.end, token_1.end), "There"),
                (Span(token_3.begin, token_3.end), "Last"),
            ]

        self.assertEqual(
            list(replace_list[data_pack.pack_id]),
            expectations
        )

if __name__ == "__main__":
    unittest.main()