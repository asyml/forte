"""This module tests Machine Translation processor."""
import unittest
import os
import tempfile
import shutil

from ddt import ddt, data, unpack
from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import MultiPackSentenceReader
from forte.processors import MicrosoftBingTranslator
from ft.onto.base_ontology import Token, Sentence

@unittest.skip("BingTranslator will be moved into examples. A texar model will "
               "be used to write NMT processor.")
@ddt
class TestMachineTranslationProcessor(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @data((["Hallo, Guten Morgen",
            "Das ist Forte. Ein tool für NLP"],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx+1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline()
        reader_config = HParams({"input_pack_name": "input",
                                 "output_pack_name": "output"},
                                MultiPackSentenceReader.default_hparams())
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        translator_config = HParams(
            {"src_language": "de", "target_language": "en",
             "in_pack_name": "input", "out_pack_name": "result"}, None)

        nlp.add_processor(MicrosoftBingTranslator(),
                          config=translator_config)
        nlp.initialize()

        english_results = ["Hey good morning", "This is Forte. A tool for NLP"]
        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            self.assertEqual(set(m_pack._pack_names),
                             set(["input", "output", "result"]))
            self.assertEqual(m_pack.get_pack("result").text,
                             english_results[idx] + "\n")
