# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit tests for Machine Translation processor.
"""
import unittest
import os
import tempfile
import shutil

from ddt import ddt, data, unpack

from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.data.readers import MultiPackSentenceReader
from forte.processors.third_party import MicrosoftBingTranslator


@unittest.skip(
    "BingTranslator will be moved into examples. A texar model will "
    "be used to write NMT processor."
)
@ddt
class TestMachineTranslationProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @data((["Hallo, Guten Morgen", "Das ist Forte. Ein tool für NLP"],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx + 1}.txt")
            with open(file_path, "w") as f:
                f.write(text)

        nlp = Pipeline[MultiPack]()
        reader_config = {
            "input_pack_name": "input",
            "output_pack_name": "output",
        }
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        translator_config = {
            "src_language": "de",
            "target_language": "en",
            "in_pack_name": "input",
            "out_pack_name": "result",
        }

        nlp.add(MicrosoftBingTranslator(), config=translator_config)
        nlp.initialize()

        english_results = ["Hey good morning", "This is Forte. A tool for NLP"]
        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            self.assertEqual(
                set(m_pack._pack_names), set(["input", "output", "result"])
            )
            self.assertEqual(
                m_pack.get_pack("result").text, english_results[idx] + "\n"
            )


if __name__ == "__main__":
    unittest.main()
