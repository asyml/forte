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
Unit tests for Deserialize Reader.
"""
import unittest

from forte.data.readers import StringReader, DeserializeReader
from forte.pipeline import Pipeline


class DeserializeReaderPipelineTest(unittest.TestCase):

    def setUp(self):

        # Define and config the Pipeline
        self.nlp = Pipeline()
        self.nlp.set_reader(StringReader())
        self.nlp.initialize()

    def test_process_next(self):

        another_pipeline = Pipeline()
        another_pipeline.set_reader(DeserializeReader())
        another_pipeline.initialize()

        data = ["Testing Reader", "Testing Deserializer"]

        for pack in self.nlp.process_dataset(data):
            for new_pack in another_pipeline.process_dataset([pack.serialize()]):
                self.assertEqual(pack.text, new_pack.text)


if __name__ == '__main__':
    unittest.main()
