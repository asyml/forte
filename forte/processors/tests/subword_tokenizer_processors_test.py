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
Unit tests for Subword Tokenizers.
"""

import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.processors.subword_tokenizer_processors import BERTTokenizer
from forte.utils.test import pretrained_test
from ft.onto.base_ontology import Subword, Sentence


class TestBERTTokenizer(unittest.TestCase):

    def setUp(self):
        self.pipeline = Pipeline()
        self.pipeline.set_reader(StringReader())
        self.pipeline.add_processor(NLTKSentenceSegmenter())
        self.pipeline.add_processor(BERTTokenizer())
        self.pipeline.initialize()

    @pretrained_test
    def test_tokenizer(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        subwords = [["This", "tool", "is", "called", "Forte", "."],
                    ["The", "goal", "of", "this", "project", "to", "help",
                     "you", "build", "NL", "P", "pipeline", "s", "."],
                    ["NL", "P", "has", "never", "been", "made", "this", "easy",
                     "before", "."]]
        document = ' '.join(sentences)
        pack = self.pipeline.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            for j, subword in enumerate(
                    pack.get(entry_type=Subword, range_annotation=sentence)):
                self.assertEqual(subword.text, subwords[i][j])
