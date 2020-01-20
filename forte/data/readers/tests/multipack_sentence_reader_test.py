import os
import tempfile
import unittest

from ddt import ddt, data, unpack
from texar.torch import HParams

from forte.data import MultiPack
from forte.data.readers import MultiPackSentenceReader
from forte.pipeline import Pipeline


@ddt
class MultiPackSentenceReaderTest(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    @data(("This file is used for testing MultiPackSentenceReader.", 1),
          ("This tool is called Forte.\n"
           "The goal of this project to help you build NLP pipelines.\n"
           "NLP has never been made this easy before.", 3))
    @unpack
    def test_parse_pack(self, text, annotation_length):

        file_path = os.path.join(self.test_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write(text)

        multipack = list(MultiPackSentenceReader().parse_pack(
            (self.test_dir, file_path)))[0]
        input_pack = multipack.get_pack('input_src')
        self.assertEqual(len(multipack.packs), 2)
        self.assertEqual(multipack._pack_names, ['input_src', 'output_tgt'])
        self.assertEqual(len(input_pack.annotations), annotation_length)
        self.assertEqual(input_pack.text, text + "\n")

    @data((["This file is used for testing MultiPackSentenceReader.",
            "This tool is called Forte. The goal of this project to help you "
            "build NLP pipelines. NLP has never been made this easy before."],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline()
        reader_config = HParams({"input_pack_name": "input",
                                 "output_pack_name": "output"},
                                MultiPackSentenceReader.default_hparams())
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        nlp.initialize()

        m_pack: MultiPack
        for m_pack in nlp.process_dataset(self.test_dir):
            # Recover the test sentence order from the doc id.
            docid = m_pack.get_pack("input").meta.doc_id
            idx = int(os.path.basename(docid).rstrip('.txt')) - 1
            self.assertEqual(m_pack._pack_names, ["input", "output"])
            self.assertEqual(m_pack.get_pack("input").text, texts[idx] + "\n")


if __name__ == "__main__":
    unittest.main()
