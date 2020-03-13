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
Unit tests for Stanford NLP processors.
"""
import os
import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.stanfordnlp_processor import StandfordNLPProcessor


class TestStanfordNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.stanford_nlp = Pipeline()
        self.stanford_nlp.set_reader(StringReader())
        models_path = os.getcwd()
        config = {
            "processors": "tokenize",
            "lang": "en",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }
        self.stanford_nlp.add_processor(StandfordNLPProcessor(models_path),
                                        config=config)
        self.stanford_nlp.initialize()

    # TODO
    @unittest.skip("We need to test this without needing to download models "
                   "everytime")
    def test_stanford_processor(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.stanford_nlp.process(document)
        print(pack)
