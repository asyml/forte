import unittest
from ddt import data, ddt

from forte.data.caster import MultiPackBoxer, MultiPackUnboxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers.misc_readers import RawPackReader, RawMultiPackReader
from forte.data.readers.multipack_sentence_reader import MultiPackSentenceReader
from forte.data.readers.multipack_terminal_reader import MultiPackTerminalReader
from forte.data.readers.plaintext_reader import PlainTextReader


@ddt
class DataPackTypeInferTest(unittest.TestCase):
    @data(
        PlainTextReader,
        RawPackReader,
    )
    def test_datapack_reader(self, component):
        reader = component()
        self.assertTrue(reader.pack_type() is DataPack)

    @data(
        MultiPackSentenceReader,
        MultiPackTerminalReader,
        RawMultiPackReader,
    )
    def test_multipack_reader(self, component):
        reader = component()
        self.assertTrue(reader.pack_type() is MultiPack)

    @data(
        MultiPackBoxer,
    )
    def test_multipack_boxer(self, component):
        caster = component()
        self.assertTrue(caster.input_pack_type() is DataPack)
        self.assertTrue(caster.output_pack_type() is MultiPack)

    @data(
        MultiPackUnboxer,
    )
    def test_multipack_unboxer(self, component):
        caster = component()
        self.assertTrue(caster.input_pack_type() is MultiPack)
        self.assertTrue(caster.output_pack_type() is DataPack)
