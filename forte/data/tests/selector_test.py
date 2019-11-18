"""
Unit tests for Selector
"""
import logging
import os
import unittest

from forte.data import DataPack, MultiPack
from forte.data.selector import NameMatchSelector, RegexNameMatchSelector, \
    FirstPackSelector, AllPackSelector

logging.basicConfig(level=logging.DEBUG)


class SelectorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_pack1 = DataPack(doc_id="1")
        self.data_pack2 = DataPack(doc_id="2")
        self.data_pack3 = DataPack(doc_id="Three")
        self.multi_pack = MultiPack()
        self.multi_pack.add_pack(self.data_pack1, pack_name="pack1")
        self.multi_pack.add_pack(self.data_pack2, pack_name="pack2")
        self.multi_pack.add_pack(self.data_pack3, pack_name="pack_three")

    def test_name_match_selector(self) -> None:
        selector = NameMatchSelector(select_name="pack1")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.meta.doc_id)

    def test_regex_name_match_selector(self) -> None:
        selector = RegexNameMatchSelector(select_name="^.*\\d$")
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.meta.doc_id)

    def test_first_pack_selector(self) -> None:
        selector = FirstPackSelector()
        packs = list(selector.select(self.multi_pack))
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0].meta.doc_id, "1")

    def test_all_pack_selector(self) -> None:
        selector = AllPackSelector()
        packs = selector.select(self.multi_pack)
        doc_ids = ["1", "2", "Three"]
        for doc_id, pack in zip(doc_ids, packs):
            self.assertEqual(doc_id, pack.meta.doc_id)
