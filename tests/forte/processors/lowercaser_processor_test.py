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
Unit tests for LowerCaser processor.
"""
import unittest
from testfixtures import LogCapture, log_capture

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.misc import LowerCaserProcessor


class TestLowerCaserProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())
        self.nlp.add(LowerCaserProcessor())
        self.nlp.initialize()

    def test_lowercaser_processor(self):
        document = (
            "This tool is called Forte. The goal of this project to "
            "help you build NLP pipelines. NLP has never been made "
            "this easy before."
        )
        pack = self.nlp.process(document)
        assert (
            pack.text == "this tool is called forte. the goal of this "
            "project to help you build nlp pipelines. nlp "
            "has never been made this easy before."
        )

    @log_capture()
    def test_text_cannot_be_lowercased(self):
        document = "Yıldız İbrahimova"
        with LogCapture() as lc:
            pack = self.nlp.process(document)
            logger, log_type, msg = lc.actual()[0]

            # Make sure the logging error is correct.
            expected_log = (
                f"Some characters cannot be converted to lower "
                f"case without changing length in pack"
            )
            assert expected_log in msg

            # Make sure the data pack is not changed.
            self.assertEqual(pack.text, document)

    def test_lowercase_with_substitution(self):
        document = "Yıldız İbrahimova"
        pack = (
            Pipeline[DataPack]()
            .set_reader(StringReader())
            .add(
                LowerCaserProcessor(),
                config={"custom_substitutions": {"İ": "i"}},
            )
            .initialize()
            .process(document)
        )

        self.assertNotEqual(pack.text, document)


if __name__ == "__main__":
    unittest.main()
