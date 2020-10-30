#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest

from typing import Dict, Any

import torch
from torch import Tensor, nn
from torch.optim import SGD

from forte.data.extractor.feature import Feature
from forte.data.extractor.converter import Converter
from forte.data.extractor.train_pipeline import TrainPipeline
from forte.data.extractor.trainer import Trainer
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.extractor import TextExtractor, CharExtractor, \
    AnnotationSeqExtractor, BaseExtractor
from ft.onto.base_ontology import Sentence, Token, EntityMention


class TrainPipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "max_char_length": 45,
            "train_path": "../../../../data_samples/train_pipeline_test",
            "val_path": "../../../../data_samples/train_pipeline_test",
            "num_epochs": 1,
            "batch_size_tokens": 10,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True
        }

        self.data_request = {
            "scope": Sentence,
            "schemes": {
                "text_tag": {
                    "entry": Token,
                    "repr": "text_repr",
                    "conversion_method": "indexing",
                    "vocab_use_pad": True,
                    "extractor": TextExtractor
                },
                "char_tag": {
                    "entry": Token,
                    "repr": "char_repr",
                    "conversion_method": "indexing",
                    "max_char_length": self.config['max_char_length'],
                    "vocab_use_pad": True,
                    "extractor": CharExtractor
                },
                "ner_tag": {
                    "entry": EntityMention,
                    "attribute": "ner_type",
                    "based_on": Token,
                    "strategy": "BIO",
                    "conversion_method": "indexing",
                    "vocab_use_pad": True,
                    "extractor": AnnotationSeqExtractor
                }
            }
        }

        def create_model_fn(schemes: Dict[str, Dict[str, Any]]):
            pass

        def create_optim_fn(model):
            pass

        def pass_tensor_to_model_fn(model,
                                    tensors: Dict[str, Dict[str, Tensor]]):
            pass

        self.create_model_fn = create_model_fn
        self.create_optim_fn = create_optim_fn
        self.pass_tensor_to_model_fn = pass_tensor_to_model_fn

        self.reader = CoNLL03Reader()

        self.trainer = Trainer(create_model_fn=create_model_fn,
                               create_optim_fn=create_optim_fn,
                               pass_tensor_to_model_fn=pass_tensor_to_model_fn)

        self.train_pipeline = \
            TrainPipeline(train_reader=self.reader,
                          dev_reader=self.reader,
                          trainer=self.trainer,
                          train_path=self.config["train_path"],
                          evaluator=None,
                          val_path=self.config["val_path"],
                          num_epochs=self.config["num_epochs"],
                          batch_size=self.config["batch_size_tokens"])

        # TODO: calculate expected loss
        self.expected_loss = ...

    def test_parse_request(self):
        self.train_pipeline._parse_request(self.data_request)
        self.assertTrue(self.train_pipeline.resource is not None)
        self.assertTrue("scope" in self.train_pipeline.resource)
        self.assertTrue("schemes" in self.train_pipeline.resource)

        self.assertTrue(len(self.train_pipeline.resource["schemes"]), 3)
        self.assertTrue("text_tag" in self.train_pipeline.resource["schemes"])
        self.assertTrue("char_tag" in self.train_pipeline.resource["schemes"])
        self.assertTrue("ner_tag" in self.train_pipeline.resource["schemes"])

        for tag, scheme in self.train_pipeline.resource["schemes"].items():
            self.assertTrue("extractor" in scheme)
            self.assertTrue("converter" in scheme)
            self.assertTrue(issubclass(type(scheme["extractor"]),
                                       BaseExtractor))
            self.assertTrue(isinstance(scheme["converter"], Converter))

        # TODO: test invalid request

    def test_build_vocab(self):
        self.train_pipeline._parse_request(self.data_request)

        self.train_pipeline._build_vocab()

        schemes: Dict[str, Any] = self.train_pipeline.resource["schemes"]

        text_extractor: TextExtractor = schemes["text_tag"]["extractor"]
        self.assertTrue(text_extractor.contains("EU"))
        self.assertTrue(text_extractor.contains("Peter"))

        char_extractor: CharExtractor = schemes["char_tag"]["extractor"]
        self.assertTrue(char_extractor.contains("a"))
        self.assertTrue(char_extractor.contains("b"))
        self.assertTrue(char_extractor.contains("."))

        ner_extractor: AnnotationSeqExtractor = schemes["ner_tag"]["extractor"]
        self.assertTrue(ner_extractor.contains(("PER", "B")))
        self.assertTrue(ner_extractor.contains((None, "O")))
        self.assertTrue(ner_extractor.contains(("MISC", "I")))

    def test_extract(self):
        self.train_pipeline._parse_request(self.data_request)
        self.train_pipeline._build_vocab()

        data_pack = list(self.reader.iter(self.train_pipeline.train_path))[0]
        feature_list = self.train_pipeline._extract(data_pack)

        self.assertEqual(len(feature_list), 7)
        for feature in feature_list:
            self.assertTrue("text_tag" in feature)
            self.assertEqual(type(feature["text_tag"]), Feature)
            self.assertTrue("char_tag" in feature)
            self.assertEqual(type(feature["char_tag"]), Feature)
            self.assertTrue("ner_tag" in feature)
            self.assertEqual(type(feature["ner_tag"]), Feature)

    def test_batch(self):
        self.train_pipeline.config.batch_size = 5
        self.train_pipeline._parse_request(self.data_request)
        self.train_pipeline._build_vocab()

        data_pack = list(self.reader.iter(self.train_pipeline.train_path))[0]
        feature_list = self.train_pipeline._extract(data_pack)

        actual_batch_num = 0
        for batch_feature_collection in \
                self.train_pipeline._batch(feature_list):
            self.assertTrue("text_tag" in batch_feature_collection)
            self.assertTrue("char_tag" in batch_feature_collection)
            self.assertTrue("ner_tag" in batch_feature_collection)

            self.assertTrue(len(batch_feature_collection["text_tag"]), 5)
            self.assertTrue(len(batch_feature_collection["char_tag"]), 5)
            self.assertTrue(len(batch_feature_collection["ner_tag"]), 5)

            actual_batch_num += 1

        self.assertEqual(actual_batch_num, 2)

    def test_convert(self):
        self.train_pipeline._parse_request(self.data_request)
        self.train_pipeline._build_vocab()

        data_pack = list(self.reader.iter(self.train_pipeline.train_path))[0]
        feature_list = self.train_pipeline._extract(data_pack)

        for batch_feature_collection in \
                self.train_pipeline._batch(feature_list):

            batch_tensor_collection = \
                 self.train_pipeline._convert(batch_feature_collection)

            self.assertTrue("text_tag" in batch_tensor_collection)
            self.assertTrue("char_tag" in batch_tensor_collection)
            self.assertTrue("ner_tag" in batch_tensor_collection)

            batch_text_tensor = batch_tensor_collection["text_tag"]
            self.assertTrue("tensor" in batch_text_tensor)
            self.assertEqual(type(batch_text_tensor["tensor"]), torch.Tensor)
            self.assertTrue("mask" in batch_text_tensor)
            self.assertEqual(len(batch_text_tensor["mask"]), 1)
            self.assertEqual(type(batch_text_tensor["mask"][0]), torch.Tensor)

            batch_char_tensor = batch_tensor_collection["char_tag"]
            self.assertTrue("tensor" in batch_char_tensor)
            self.assertEqual(type(batch_char_tensor["tensor"]), torch.Tensor)
            self.assertTrue("mask" in batch_char_tensor)
            self.assertEqual(len(batch_char_tensor["mask"]), 2)
            self.assertEqual(type(batch_char_tensor["mask"][0]), torch.Tensor)
            self.assertEqual(type(batch_char_tensor["mask"][1]), torch.Tensor)

            batch_ner_tensor = batch_tensor_collection["ner_tag"]
            self.assertTrue("tensor" in batch_ner_tensor)
            self.assertEqual(type(batch_ner_tensor["tensor"]), torch.Tensor)
            self.assertTrue("mask" in batch_ner_tensor)
            self.assertEqual(len(batch_ner_tensor["mask"]), 1)
            self.assertEqual(type(batch_ner_tensor["mask"][0]), torch.Tensor)

    # TODO: add a test for testing TrainPipeline::run


if __name__ == '__main__':
    unittest.main()
