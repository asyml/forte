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
Unit tests for pretrained embedders.
"""

import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.processors.pretrained_encoder_processors import PretrainedEncoder
from forte.utils.test import pretrained_test
from ft.onto.base_ontology import Sentence


class TestBERTEmbedder(unittest.TestCase):

    def setUp(self):
        self.pipeline = Pipeline()
        self.pipeline.set_reader(StringReader())
        self.pipeline.add_processor(NLTKSentenceSegmenter())
        self.pipeline.add_processor(PretrainedEncoder())
        self.pipeline.initialize()

    @pretrained_test
    def test_embedder(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.pipeline.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(sentence.embedding.shape, (1, 512, 768))


if __name__ == "__main__":
    unittest.main()
