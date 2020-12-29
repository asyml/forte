#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
import unittest
import torch
from typing import Dict, Any, Iterator

from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from forte.data.base_pack import PackType
from forte.data.vocabulary import Vocabulary
from forte.data.converter import Converter
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.attribute_extractor import AttributeExtractor
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.extractor.char_extractor import CharExtractor
from forte.data.extractor.seqtagging_extractor import BioSeqTaggingExtractor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Token, EntityMention


class TrainPreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "max_char_length": 45,
            "train_path": "data_samples/train_pipeline_test",
            "val_path": "data_samples/train_pipeline_test",
            "num_epochs": 1,
            "batch_size_tokens": 5,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True
        }

        text_extractor: AttributeExtractor = \
            AttributeExtractor(config={"entry_type": Token,
                                       "vocab_method": "indexing",
                                       "attribute_get": "text"})

        char_extractor: CharExtractor = \
            CharExtractor(
                config={"entry_type": Token,
                        "vocab_method": "indexing",
                        "max_char_length": self.config["max_char_length"]})

        # Add output part in request based on different task type
        ner_extractor: BioSeqTaggingExtractor = \
            BioSeqTaggingExtractor(config={"entry_type": EntityMention,
                                           "attribute": "ner_type",
                                           "based_on": Token,
                                           "vocab_method": "indexing"})

        self.tp_request = {
            "scope": Sentence,
            "schemes": {
                "text_tag": {
                    "type": TrainPreprocessor.DATA_INPUT,
                    "extractor": text_extractor
                },
                "char_tag": {
                    "type": TrainPreprocessor.DATA_INPUT,
                    "extractor": char_extractor
                },
                "ner_tag": {
                    "type": TrainPreprocessor.DATA_OUTPUT,
                    "extractor": ner_extractor
                }
            }
        }

        self.reader = CoNLL03Reader()

        self.evaluator = CoNLLNEREvaluator()

        self.tp_config = {
            "dataset": {
                "batch_size": self.config["batch_size_tokens"]
            }
        }

        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(self.reader)
        train_pl.initialize()
        pack_generator: Iterator[PackType] = \
            train_pl.process_dataset(self.config["train_path"])

        self.train_preprocessor = \
            TrainPreprocessor(pack_generator=pack_generator,
                              request=self.tp_request,
                              config=self.tp_config)

    def test_parse_request(self):
        self.assertTrue(self.train_preprocessor.request is not None)
        self.assertTrue("scope" in self.train_preprocessor.request)
        self.assertTrue("schemes" in self.train_preprocessor.request)

        self.assertTrue(
            len(self.train_preprocessor.request["schemes"]), 3)
        self.assertTrue(
            "text_tag" in self.train_preprocessor.request["schemes"])
        self.assertTrue(
            "char_tag" in self.train_preprocessor.request["schemes"])
        self.assertTrue(
            "ner_tag" in self.train_preprocessor.request["schemes"])

        for tag, scheme in \
                self.train_preprocessor.request["schemes"].items():
            self.assertTrue("extractor" in scheme)
            self.assertTrue("converter" in scheme)
            self.assertTrue(issubclass(type(scheme["extractor"]),
                                       BaseExtractor))
            self.assertTrue(isinstance(scheme["converter"], Converter))

    def test_build_vocab(self):
        schemes: Dict[str, Any] = \
            self.train_preprocessor.request["schemes"]

        text_extractor: AttributeExtractor = schemes["text_tag"]["extractor"]
        vocab: Vocabulary = text_extractor.vocab
        self.assertTrue(vocab.has_element("EU"))
        self.assertTrue(vocab.has_element("Peter"))

        char_extractor: CharExtractor = schemes["char_tag"]["extractor"]
        vocab: Vocabulary = char_extractor.vocab
        self.assertTrue(vocab.has_element("a"))
        self.assertTrue(vocab.has_element("b"))
        self.assertTrue(vocab.has_element("."))

        ner_extractor: BioSeqTaggingExtractor = schemes["ner_tag"]["extractor"]
        vocab: Vocabulary = ner_extractor.vocab
        self.assertTrue(vocab.has_element(("PER", "B")))
        self.assertTrue(vocab.has_element((None, "O")))
        self.assertTrue(vocab.has_element(("MISC", "I")))

    def test_build_dataset_iterator(self):
        train_iterator = \
            self.train_preprocessor._build_dataset_iterator()

        batchs = []
        for batch in train_iterator:
            batchs.append(batch)

        self.assertEqual(len(batchs), 2)
        self.assertEqual(batchs[0].batch_size, 5)
        self.assertEqual(batchs[1].batch_size, 2)

        for batch in batchs:
            self.assertTrue(hasattr(batch, "text_tag"))
            self.assertTrue(hasattr(batch, "char_tag"))
            self.assertTrue(hasattr(batch, "ner_tag"))

            for tag, batch_t in batch.items():
                self.assertTrue("data" in batch_t)
                self.assertEqual(type(batch_t["data"]), torch.Tensor)
                self.assertTrue("masks" in batch_t)
                if tag == "text_tag" or tag == "ner_tag":
                    self.assertEqual(len(batch_t["masks"]), 1)
                    self.assertEqual(type(batch_t["masks"][0]), torch.Tensor)
                else:
                    self.assertEqual(len(batch_t["masks"]), 2)
                    self.assertEqual(type(batch_t["masks"][0]), torch.Tensor)
                    self.assertEqual(type(batch_t["masks"][1]), torch.Tensor)


if __name__ == '__main__':
    unittest.main()
