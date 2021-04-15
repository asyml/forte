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
Unit tests for NLTK processors.
"""

import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.third_party import NLTKSentenceSegmenter
from forte.processors.third_party import VaderSentimentProcessor
from ft.onto.base_ontology import Sentence


class TestVaderSentiment(unittest.TestCase):

    def setUp(self):
        self.pipeline = Pipeline[DataPack]()
        self.pipeline.set_reader(StringReader())
        self.pipeline.add(NLTKSentenceSegmenter())
        self.pipeline.add(VaderSentimentProcessor())
        self.pipeline.initialize()

    def test_segmenter(self):
        sentences = [
            "VADER is smart, handsome, and funny.",
            # positive sentence example
            "VADER is smart, handsome, and funny!",
            # punctuation emphasis handled correctly (sentiment intensity
            # adjusted)
            "VADER is very smart, handsome, and funny.",
            # booster words handled correctly (sentiment intensity adjusted)
            "VADER is VERY SMART, handsome, and FUNNY.",
            # emphasis for ALLCAPS handled
            "VADER is VERY SMART, handsome, and FUNNY!!!",
            # combination of signals - VADER appropriately adjusts intensity
            "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
            # booster words & punctuation make this close to ceiling for score
            "VADER is not smart, handsome, nor funny.",
            # negation sentence example
            "The book was good.",  # positive sentence
            "At least it isn't a horrible book.",
            # negated negative sentence with contraction
            "The book was only kind of good.",
            # qualified positive sentence is handled correctly (intensity
            # adjusted)
            "The plot was good, but the characters are uncompelling and the "
            "dialog is not great.",
            # mixed negation sentence
            "Today SUX!",
            # negative slang with capitalization emphasis
            "Today only kinda sux! But I'll get by, lol",
            # mixed sentiment example with slang and constrastive conjunction
            # "but"
            "Make sure you :) or :D today!",  # emoticons handled
            "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",
            # emojis handled
            "Not bad at all"  # Capitalized negation
        ]

        expected_scores = [
            {'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8316},
            {'neg': 0.0, 'neu': 0.248, 'pos': 0.752, 'compound': 0.8439},
            {'neg': 0.0, 'neu': 0.299, 'pos': 0.701, 'compound': 0.8545},
            {'neg': 0.0, 'neu': 0.246, 'pos': 0.754, 'compound': 0.9227},
            {'neg': 0.0, 'neu': 0.233, 'pos': 0.767, 'compound': 0.9342},
            {'neg': 0.0, 'neu': 0.294, 'pos': 0.706, 'compound': 0.9469},
            {'neg': 0.646, 'neu': 0.354, 'pos': 0.0, 'compound': -0.7424},
            {'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404},
            {'neg': 0.0, 'neu': 0.678, 'pos': 0.322, 'compound': 0.431},
            {'neg': 0.0, 'neu': 0.697, 'pos': 0.303, 'compound': 0.3832},
            {'neg': 0.327, 'neu': 0.579, 'pos': 0.094, 'compound': -0.7042},
            {'neg': 0.779, 'neu': 0.221, 'pos': 0.0, 'compound': -0.5461},
            {'neg': 0.454, 'neu': 0.546, 'pos': 0.0, 'compound': -0.3609},
            {'neg': 0.0, 'neu': 0.327, 'pos': 0.673, 'compound': 0.9551},
            {'neg': 0.0, 'neu': 0.698, 'pos': 0.302, 'compound': 0.8248},
        ]

        expected_categories = [
            s['compound'] > 0 for s in expected_scores
        ]

        document = ' '.join(sentences)
        pack = self.pipeline.process(document)

        # testing only polarity of the scores as the exact scores depend on the
        # version of sentimentVader
        expected_categories = [s['compound'] > 0 for s in expected_scores]

        sentence: Sentence
        for idx, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(sentence.sentiment['compound'] > 0,
                             expected_categories[idx])


if __name__ == "__main__":
    unittest.main()
