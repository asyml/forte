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
Unit tests for MultiPackSentenceReader.
"""
import os
import tempfile
import unittest

from ddt import ddt, data, unpack
from texar.torch import HParams

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

        multipack = list(MultiPackSentenceReader().parse_pack(file_path))[0]
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
            file_path = os.path.join(self.test_dir, f"{idx+1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline()
        reader_config = {"input_pack_name": "input",
                         "output_pack_name": "output"}
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            self.assertEqual(m_pack._pack_names, ["input", "output"])
            self.assertEqual(m_pack.get_pack("input").text, texts[idx] + "\n")


if __name__ == "__main__":
    unittest.main()
