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
Unit tests for pretrained encoders.
"""

import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKChunker
from forte.processors.pretrained_encoder_processors import PretrainedEncoder
from forte.utils.test import pretrained_test
from ft.onto.base_ontology import Document, Phrase, Sentence


class TestPretrainedEncoder(unittest.TestCase):

    @pretrained_test
    def test_encoder_sentence(self):
        pipeline = Pipeline()
        pipeline.set_reader(StringReader())
        pipeline.add_processor(NLTKSentenceSegmenter())
        pipeline.add_processor(PretrainedEncoder())
        pipeline.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = pipeline.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(sentence.embedding.shape, (1, 512, 768))

    @pretrained_test
    def test_encoder_document(self):
        pipeline = Pipeline()
        pipeline.set_reader(StringReader())
        pipeline.add_processor(
            PretrainedEncoder(),
            config={'entry_type': 'ft.onto.base_ontology.Document'})
        pipeline.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = pipeline.process(document)
        for i, doc in enumerate(pack.get(Document)):
            self.assertEqual(doc.embedding.shape, (1, 512, 768))

    @pretrained_test
    def test_encoder_phrase(self):
        pipeline = Pipeline()
        pipeline.set_reader(StringReader())
        pipeline.add_processor(NLTKSentenceSegmenter())
        pipeline.add_processor(NLTKWordTokenizer())
        pipeline.add_processor(NLTKPOSTagger())
        config = {'pattern': 'NP: {<DT>?<JJ>*<NN>}'}
        pipeline.add_processor(NLTKChunker(), config=config)
        pipeline.add_processor(
            PretrainedEncoder(),
            config={'entry_type': 'ft.onto.base_ontology.Phrase'})
        pipeline.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = pipeline.process(document)
        for i, phrase in enumerate(pack.get(Phrase)):
            self.assertEqual(phrase.embedding.shape, (1, 512, 768))


if __name__ == "__main__":
    unittest.main()
