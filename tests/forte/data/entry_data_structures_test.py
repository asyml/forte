import unittest
from dataclasses import dataclass
from typing import Optional, List, Any, Iterator

from forte.data.base_reader import PackReader, MultiPackReader
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Generics, MultiPackGeneric, Annotation
from forte.data.ontology.core import FList, FDict, Pointer
from forte.pipeline import Pipeline
from forte.processors.base import PackProcessor, MultiPackProcessor
from ft.onto.base_ontology import EntityMention


@dataclass
class ExampleEntry(Generics):
    secret_number: Optional[int] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)


@dataclass
class DifferentEntry(Generics):
    secret_number: Optional[int] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)


@dataclass
class ExampleMPEntry(MultiPackGeneric):
    refer_entry: Optional[ExampleEntry] = None

    def __init__(self, pack: MultiPack):
        super().__init__(pack)


@dataclass
class EntryWithList(Generics):
    """
    Test whether entries are stored correctly as a List using FList.
    """

    entries: FList[ExampleEntry] = None

    def __init__(self, pack: MultiPack):
        super().__init__(pack)
        self.entries = FList[ExampleEntry](self)


class EntryWithDict(Generics):
    """
    Test whether entries are stored correctly as a Dict using FDict.
    """

    entries: FDict[str, ExampleEntry] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.entries = FDict[str, ExampleEntry](self)


class EntryAsAttribute(Generics):
    """
    Test whether entries are stored correctly in the entry.
    """

    att_entry: Optional[ExampleEntry] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)


class EntryWithDictAndPointer(EntryWithDict):
    another_dict_entry: FDict[str, ExampleEntry] = None
    pointer_entry: Optional[ExampleEntry] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.another_dict_entry = FDict[str, ExampleEntry](self)
        self.entries = FDict[str, ExampleEntry](self)
        self.pointer_entry = ExampleEntry(pack)


class EmptyReader(PackReader):
    def _collect(self, names: List[str]) -> Iterator[Any]:
        yield from names

    def _parse_pack(self, name: str) -> Iterator[DataPack]:
        p = DataPack()
        p.pack_name = name
        yield p


class EntryAnnotator(PackProcessor):
    def _process(self, input_pack: DataPack):
        # Add a EntryWithList
        le: EntryWithList = EntryWithList(input_pack)
        # Add a couple entry to the list.
        for i in range(10):
            e = ExampleEntry(input_pack)
            e.secret_number = i
            le.entries.append(e)

        # Add a EntryWithDict
        de: EntryWithDict = EntryWithDict(input_pack)
        for i in range(10, 20):
            e = ExampleEntry(input_pack)
            e.secret_number = i
            de.entries[str(i)] = e

        # Add a EntryWithEntry
        e_with_a: EntryAsAttribute = EntryAsAttribute(input_pack)
        ee = ExampleEntry(input_pack)
        e_with_a.att_entry = ee
        ee.secret_number = 27


class ChildEntryAnnotator(PackProcessor):
    def _process(self, input_pack: DataPack):
        e: EntryWithDictAndPointer = EntryWithDictAndPointer(input_pack)
        temp = ExampleEntry(input_pack)
        e.entries["1"] = temp
        e.another_dict_entry["2"] = temp
        e.pointer_entry = ExampleEntry(input_pack)


class EmptyMultiReader(MultiPackReader):
    def _collect(self, names: List[str]) -> Iterator[Any]:
        yield from names

    def _parse_pack(self, name: str) -> Iterator[MultiPack]:
        p = MultiPack()
        p.pack_name = name
        yield p


class MultiPackEntryAnnotator(MultiPackProcessor):
    def _process(self, multi_pack: MultiPack):
        # Add a pack.
        p1 = multi_pack.add_pack("pack1")
        p2 = multi_pack.add_pack("pack2")

        # Add some entries into one pack.
        e1: ExampleEntry = p1.add_entry(ExampleEntry(p1))
        e1.secret_number = 1
        p2.add_entry(ExampleEntry(p2))

        # Add the multi pack entry.
        mp_entry = ExampleMPEntry(multi_pack)
        mp_entry.refer_entry = e1


class MultiEntryStructure(unittest.TestCase):
    def setUp(self):
        p: Pipeline[MultiPack] = Pipeline[MultiPack]()
        p.set_reader(EmptyMultiReader())
        p.add(MultiPackEntryAnnotator())
        p.initialize()
        self.pack: MultiPack = p.process(["doc1", "doc2"])

    def test_entry_attribute_mp_pointer(self):
        mpe: ExampleMPEntry = self.pack.get_single(ExampleMPEntry)
        self.assertIsInstance(mpe.refer_entry, ExampleEntry)
        self.assertIsInstance(mpe.__dict__["refer_entry"], ExampleEntry)

        serialized_mp = self.pack.to_string(drop_record=True)
        recovered_mp = MultiPack.from_string(serialized_mp)

        s_packs = [p.to_string() for p in self.pack.packs]
        recovered_packs = [DataPack.from_string(s) for s in s_packs]

        recovered_mp.relink(recovered_packs)

        re_mpe: ExampleMPEntry = recovered_mp.get_single(ExampleMPEntry)
        self.assertIsInstance(re_mpe.refer_entry, ExampleEntry)
        self.assertEqual(re_mpe.refer_entry.tid, mpe.refer_entry.tid)
        self.assertEqual(re_mpe.tid, mpe.tid)


class EntryDataStructure(unittest.TestCase):
    def setUp(self):
        p: Pipeline = Pipeline()
        p.set_reader(EmptyReader())
        p.add(EntryAnnotator())
        p.initialize()

        self.pack: DataPack = p.process(["doc1", "doc2"])

    def test_entry_attribute(self):
        entry_with_attr: EntryAsAttribute = self.pack.get_single(
            EntryAsAttribute
        )

        # Make sure we can get the entry of correct type and data.
        self.assertIsInstance(entry_with_attr.att_entry, ExampleEntry)
        self.assertEqual(entry_with_attr.att_entry.secret_number, 27)
        self.assertIsInstance(
            entry_with_attr.__dict__["att_entry"], ExampleEntry
        )

        # Make sure the recovered entry is also correct.
        pack_str = self.pack.to_string(True)
        recovered = self.pack.from_string(pack_str)
        self.assertEqual(
            recovered.get_single(EntryAsAttribute), entry_with_attr
        )

    def test_entry_list(self):
        list_entry: EntryWithList = self.pack.get_single(EntryWithList)
        # Make sure the list data types are correct.
        for e in list_entry.entries:
            self.assertIsInstance(e, ExampleEntry)
        # Check size.
        self.assertEqual(len(list_entry.entries), 10)

        # Make sure we stored index instead of raw data in list.
        for v in list_entry.entries.__dict__["_FList__data"]:
            self.assertIsInstance(v, Pointer)

        # Make sure the recovered entry is also correct.
        pack_str = self.pack.to_string(True)
        recovered = self.pack.from_string(pack_str)

        origin_list = list_entry.entries
        recovered_list = recovered.get_single(EntryWithList).entries

        self.assertEqual(origin_list, recovered_list)

        self.assertEqual(recovered.get_single(EntryWithList), list_entry)

    def test_entry_dict(self):
        first_dict_entry: EntryWithDict = self.pack.get_single(EntryWithDict)

        # Make sure the dict data types are correct.
        for e in first_dict_entry.entries.values():
            self.assertTrue(isinstance(e, ExampleEntry))
        self.assertEqual(len(first_dict_entry.entries), 10)

        # Make sure we stored index (pointers) instead of raw data in dict.
        for v in first_dict_entry.entries.__dict__["_FDict__data"].values():
            self.assertTrue(isinstance(v, Pointer))

        # Make sure the recovered entry is also correct.
        pack_str = self.pack.to_string(True)
        recovered = DataPack.from_string(pack_str)

        recovered_first = recovered.get_single(EntryWithDict)

        self.assertTrue("10" in recovered_first.entries)

        self.assertEqual(recovered_first, first_dict_entry)

        self.assertEqual(recovered_first.entries, first_dict_entry.entries)

    def test_entry_key_memories(self):
        pack = (
            Pipeline[MultiPack]()
            .set_reader(EmptyReader())
            .add(ChildEntryAnnotator())
            .initialize()
            .process(["pack1", "pack2"])
        )

        DataPack.from_string(pack.to_string(True))

        from forte.data.ontology import core

        self.assertTrue(
            core._f_struct_keys[
                "entry_data_structures_test.EntryWithDict_entries"
            ]
        )
        self.assertNotIn(
            "entry_data_structures_test.EntryWithDict_another_dict_entry",
            core._f_struct_keys,
        )
        self.assertNotIn(
            "entry_data_structures_test.EntryWithDict_pointer_entry",
            core._pointer_keys,
        )

        self.assertTrue(
            core._f_struct_keys[
                "entry_data_structures_test."
                "EntryWithDictAndPointer_another_dict_entry"
            ]
        )
        self.assertTrue(
            core._pointer_keys[
                "entry_data_structures_test."
                "EntryWithDictAndPointer_pointer_entry"
            ]
        )


class NotHashingTest(unittest.TestCase):
    def setUp(self):
        self.pack: DataPack = DataPack()
        self.pack.set_text("Some text to test annotations on.")

    def test_not_hashable(self):
        anno: Annotation = Annotation(self.pack, 0, 5)
        with self.assertRaises(TypeError):
            hash(anno)
        anno.regret_creation()

        anno1: EntityMention = EntityMention(self.pack, 0, 2)
        with self.assertRaises(TypeError):
            hash(anno1)
        anno1.regret_creation()
