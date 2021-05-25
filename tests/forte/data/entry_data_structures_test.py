import unittest
from dataclasses import dataclass
from typing import Optional, List, Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Generics, MultiPackGeneric, Annotation
from forte.data.ontology.core import FList, FDict, MpPointer, Pointer
from forte.data.base_reader import PackReader, MultiPackReader
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
    entries: FDict[int, ExampleEntry] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.entries = FDict[int, ExampleEntry](self)


class EntryAsAttribute(Generics):
    """
    Test whether entries are stored correctly in the entry.
    """
    att_entry: Optional[ExampleEntry] = None

    def __init__(self, pack: DataPack):
        super().__init__(pack)


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
            de.entries[i] = e

        # Add a EntryWithEntry
        e_with_a: EntryAsAttribute = EntryAsAttribute(input_pack)
        ee = ExampleEntry(input_pack)
        e_with_a.att_entry = ee
        ee.secret_number = 27


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
        p1 = multi_pack.add_pack('pack1')
        p2 = multi_pack.add_pack('pack2')

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
        self.pack: MultiPack = p.process(['doc1', 'doc2'])

    def test_entry_attribute(self):
        mpe: ExampleMPEntry = self.pack.get_single(ExampleMPEntry)
        self.assertIsInstance(mpe.refer_entry, ExampleEntry)
        self.assertIsInstance(mpe.__dict__['refer_entry'], MpPointer)

    def test_wrong_attribute(self):
        import warnings
        input_pack = MultiPack()
        mp_entry = ExampleMPEntry(input_pack)
        p1 = input_pack.add_pack('pack1')
        e1: DifferentEntry = p1.add_entry(DifferentEntry(p1))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mp_entry.refer_entry = e1
            mp_entry.regret_creation()
            assert issubclass(w[-1].category, UserWarning)
            # self.assertEqual(str(w[-1].message), warning_content)


class EntryDataStructure(unittest.TestCase):
    def setUp(self):
        p: Pipeline = Pipeline()
        p.set_reader(EmptyReader())
        p.add(EntryAnnotator())
        p.initialize()

        self.pack: DataPack = p.process(['doc1', 'doc2'])

    def test_entry_attribute(self):
        entry_with_attr: EntryAsAttribute = self.pack.get_single(
            EntryAsAttribute)

        # Make sure we can get the entry of correct type and data.
        self.assertIsInstance(entry_with_attr.att_entry, ExampleEntry)
        self.assertEqual(entry_with_attr.att_entry.secret_number, 27)
        self.assertIsInstance(entry_with_attr.__dict__['att_entry'], Pointer)

    def test_entry_list(self):
        list_entry: EntryWithList = self.pack.get_single(EntryWithList)
        # Make sure the list data types are correct.
        for e in list_entry.entries:
            self.assertIsInstance(e, ExampleEntry)
        # Check size.
        self.assertEqual(len(list_entry.entries), 10)

        # Make sure we stored index instead of raw data in list.
        for v in list_entry.entries.__dict__['_FList__data']:
            self.assertIsInstance(v, Pointer)

    def test_entry_dict(self):
        dict_entry: EntryWithDict = self.pack.get_single(EntryWithDict)

        # Make sure the dict data types are correct.
        for e in dict_entry.entries.values():
            self.assertTrue(isinstance(e, ExampleEntry))
        self.assertEqual(len(dict_entry.entries), 10)

        # Make sure we stored index (pointers) instead of raw data in dict.
        for v in dict_entry.entries.__dict__['_FDict__data'].values():
            self.assertTrue(isinstance(v, Pointer))


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
