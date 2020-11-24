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

from typing import Dict, Any

from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from torch import Tensor

from forte.data.extractor.vocabulary import Vocabulary
from forte.data.types import DATA_INPUT, DATA_OUTPUT
from forte.data.converter.converter import Converter
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.extractor import TextExtractor, CharExtractor, \
    BioSeqTaggingExtractor, BaseExtractor
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

        self.tp_request = {
            "scope": Sentence,
            "schemes": {
                "text_tag": {
                    "entry_type": Token,
                    "repr": "text_repr",
                    "conversion_method": "indexing",
                    "vocab_use_pad": True,
                    "type": DATA_INPUT,
                    "extractor": TextExtractor
                },
                "char_tag": {
                    "entry_type": Token,
                    "repr": "char_repr",
                    "conversion_method": "indexing",
                    "max_char_length": self.config['max_char_length'],
                    "vocab_use_pad": True,
                    "type": DATA_INPUT,
                    "extractor": CharExtractor
                },
                "ner_tag": {
                    "entry_type": EntityMention,
                    "attribute": "ner_type",
                    "based_on": Token,
                    "strategy": "BIO",
                    "vocab_method": "indexing",
                    "vocab_use_pad": True,
                    "type": DATA_OUTPUT,
                    "extractor": BioSeqTaggingExtractor
                }
            }
        }

        self.reader = CoNLL03Reader()

        self.evaluator = CoNLLNEREvaluator()

        self.tp_config = {
            "preprocess": {
                "pack_dir": self.config["train_path"]
            },
            "dataset": {
                "batch_size": self.config["batch_size_tokens"]
            }
        }

        self.train_preprocessor = \
            TrainPreprocessor(train_reader=self.reader,
                              request=self.tp_request,
                              config=self.tp_config)

    def test_parse_request(self):
        self.assertTrue(self.train_preprocessor.feature_resource is not None)
        self.assertTrue("scope" in self.train_preprocessor.feature_resource)
        self.assertTrue("schemes" in self.train_preprocessor.feature_resource)

        self.assertTrue(
            len(self.train_preprocessor.feature_resource["schemes"]), 3)
        self.assertTrue(
            "text_tag" in self.train_preprocessor.feature_resource["schemes"])
        self.assertTrue(
            "char_tag" in self.train_preprocessor.feature_resource["schemes"])
        self.assertTrue(
            "ner_tag" in self.train_preprocessor.feature_resource["schemes"])

        for tag, scheme in \
                self.train_preprocessor.feature_resource["schemes"].items():
            self.assertTrue("extractor" in scheme)
            self.assertTrue("converter" in scheme)
            self.assertTrue(issubclass(type(scheme["extractor"]),
                                       BaseExtractor))
            self.assertTrue(isinstance(scheme["converter"], Converter))

        # TODO: test invalid request

    def test_build_vocab(self):
        schemes: Dict[str, Any] = \
            self.train_preprocessor.feature_resource["schemes"]

        text_extractor: TextExtractor = schemes["text_tag"]["extractor"]
        vocab: Vocabulary = text_extractor.vocab
        self.assertTrue(vocab.has_key("EU"))
        self.assertTrue(vocab.has_key("Peter"))

        char_extractor: CharExtractor = schemes["char_tag"]["extractor"]
        vocab: Vocabulary = char_extractor.vocab
        self.assertTrue(vocab.has_key("a"))
        self.assertTrue(vocab.has_key("b"))
        self.assertTrue(vocab.has_key("."))

        ner_extractor: BioSeqTaggingExtractor = schemes["ner_tag"]["extractor"]
        vocab: Vocabulary = ner_extractor.vocab
        self.assertTrue(vocab.has_key(("PER", "B")))
        self.assertTrue(vocab.has_key((None, "O")))
        self.assertTrue(vocab.has_key(("MISC", "I")))

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

            for tag, tensors in batch.items():
                self.assertTrue("tensor" in tensors)
                self.assertEqual(type(tensors["tensor"]), Tensor)
                self.assertTrue("mask" in tensors)
                if tag == "text_tag" or tag == "ner_tag":
                    self.assertEqual(len(tensors["mask"]), 1)
                    self.assertEqual(type(tensors["mask"][0]), Tensor)
                else:
                    self.assertEqual(len(tensors["mask"]), 2)
                    self.assertEqual(type(tensors["mask"][0]), Tensor)
                    self.assertEqual(type(tensors["mask"][1]), Tensor)


if __name__ == '__main__':
    unittest.main()
