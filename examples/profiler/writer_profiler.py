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

import os
import unittest
import tempfile

from typing import List

from forte.data.data_pack import DataPack
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.pipeline import Pipeline
from forte.processors.writers import (
    PackNameJsonPackWriter,
    PackNamePicklePackWriter
)
from forte.processors.misc import (
    PeriodSentenceSplitter
)
from forte.processors.nlp import SubwordTokenizer
import time

class OntonotesWriterPipelineTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
        # Define and config the Pipeline
        self.dataset_path = os.path.join(root_path, "Documents/forte/examples/profiler/combine_data")
        

    def test_writer(self):
        pipe_serialize = Pipeline[DataPack]()
        pipe_serialize.set_reader(OntonotesReader())

        pipe_serialize.add(PeriodSentenceSplitter())
        pipe_serialize.add(SubwordTokenizer())

        with tempfile.TemporaryDirectory() as output_dir:
            pipe_serialize.add(
                #PackNamePicklePackWriter(),
                PackNameJsonPackWriter(),
                {
                    "output_dir": output_dir,
                    "indent": 2,
                }
            )

            pipe_serialize.run(self.dataset_path)



if __name__ == "__main__":
    unittest.main('writer_profiler')
