"""
Unit tests for multi pack related operations.
"""
import logging
import unittest

from forte.data.multi_pack import MultiPack, MultiPackLink
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from ft.onto.base_ontology import (
    Token)

logging.basicConfig(level=logging.DEBUG)


def _space_token(pack: DataPack):
    begin = 0
    for i, c in enumerate(pack.text):
        if c == ' ':
            Token(pack, begin, i)
            begin = i + 1

    if begin < len(pack.text):
        Token(pack, begin, len(pack.text))


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_pack = MultiPack()
        self.data_pack1 = self.multi_pack.add_pack(pack_name="left pack")
        self.data_pack2 = self.multi_pack.add_pack(pack_name="right pack")

        self.data_pack1.doc_id = "some pack"
        self.data_pack1.set_text("This pack contains some sample data.")

        self.data_pack2.doc_id = "another pack"
        self.data_pack2.set_text("This pack contains some other sample data.")

    def test_serialization(self):
        ser_str: str = self.multi_pack.serialize()
        print(ser_str)

    def test_add_pack(self):
        data_pack3 = self.multi_pack.add_pack(pack_name="new pack")
        data_pack3.doc_id = "the third pack"
        data_pack3.set_text("Test to see if we can add new packs..")

        self.assertEqual(len(self.multi_pack.packs), 3)
        self.assertEqual(self.multi_pack.pack_names,
                         {'left pack', 'right pack', 'new pack'})

    def test_rename_pack(self):
        self.multi_pack.rename_pack('right pack', 'last pack')
        self.assertEqual(self.multi_pack.pack_names,
                         {'left pack', 'last pack'})

    def test_entry(self):
        # 1. Add tokens to each pack.
        for pack in self.multi_pack.packs:
            _space_token(pack)

        left_tokens = [t.text for t in self.multi_pack.packs[0].get(Token)]
        right_tokens = [t.text for t in self.multi_pack.packs[1].get(Token)]

        self.assertListEqual(left_tokens,
                             ["This", "pack", "contains", "some", "sample",
                              "data."])
        self.assertListEqual(right_tokens,
                             ["This", "pack", "contains", "some", "other",
                              "sample", "data."])

        # 2. Link the same words from two packs.
        token: Annotation
        left_tokens = {}
        for token in self.multi_pack.packs[0].get(Token):
            left_tokens[token.text] = token

        right_tokens = {}
        for token in self.multi_pack.packs[1].get(Token):
            right_tokens[token.text] = token

        for key, lt in left_tokens.items():
            if key in right_tokens:
                rt = right_tokens[key]
                self.multi_pack.add_entry(MultiPackLink(
                    self.multi_pack, lt, rt))

        linked_tokens = []
        for link in self.multi_pack.links:
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens,
            [("This", "This"), ("pack", "pack"), ("contains", "contains"),
             ("some", "some"), ("sample", "sample"), ("data.", "data.")])

        # 3. Test deletion
        self.multi_pack.delete_entry(self.multi_pack.links[-1])

        linked_tokens = []
        for link in self.multi_pack.links:
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_tokens.append((parent_text, child_text))

        self.assertListEqual(
            linked_tokens,
            [("This", "This"), ("pack", "pack"), ("contains", "contains"),
             ("some", "some"), ("sample", "sample")])


if __name__ == '__main__':
    unittest.main()
