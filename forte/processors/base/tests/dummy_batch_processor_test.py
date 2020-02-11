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
Unit tests for dummy processor.
"""
import unittest
from ddt import ddt, data

from forte.data.readers import OntonotesReader, StringReader, PlainTextReader
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.processors.base.tests.dummy_batch_processor import \
    DummyRelationExtractor, DummmyFixedSizeBatchProcessor
from ft.onto.base_ontology import RelationLink, Sentence


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = Pipeline()
        self.nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = {"batcher": {"batch_size": 5}}
        self.nlp.add_processor(dummy, config=config)
        self.nlp.initialize()

        self.data_path = "data_samples/ontonotes/00/"

    def test_processor(self):
        pack = self.nlp.process(self.data_path)
        relations = list(pack.get_entries(RelationLink))
        assert (len(relations) > 0)
        for relation in relations:
            self.assertEqual(relation.get_field("rel_type"), "dummy_relation")


@ddt
class DummyFixedSizeBatchProcessorTest(unittest.TestCase):

    @data(1, 2, 3)
    def test_one_batch_processor(self, batch_size):
        nlp = Pipeline()
        nlp.set_reader(StringReader())
        dummy = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add_processor(NLTKSentenceSegmenter())
        nlp.add_processor(dummy, config=config)
        nlp.initialize()
        sentences = ["This tool is called Forte. The goal of this project to "
                     "help you build NLP pipelines. NLP has never been made "
                     "this easy before."]
        pack = nlp.process(sentences)
        sent_len = len(list(pack.get(Sentence)))
        self.assertEqual(
            dummy.counter, (sent_len // batch_size +
                            (sent_len % batch_size > 0)))

    @data(1, 2, 3)
    def test_two_batch_processors(self, batch_size):
        nlp = Pipeline()
        nlp.set_reader(PlainTextReader())
        dummy1 = DummmyFixedSizeBatchProcessor()
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add_processor(NLTKSentenceSegmenter())

        nlp.add_processor(dummy1, config=config)
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add_processor(dummy2, config=config)

        nlp.initialize()
        data_path = "data_samples/random_texts"
        pack = nlp.process(data_path)
        sent_len = len(list(pack.get(Sentence)))

        self.assertEqual(
            dummy1.counter, (sent_len // batch_size +
                            (sent_len % batch_size > 0)))

        self.assertEqual(
            dummy2.counter, (sent_len // (2 * batch_size) +
                             (sent_len % (2 * batch_size) > 0)))


if __name__ == '__main__':
    unittest.main()
