# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Unit tests for MetaModel.
"""
import unittest

import torch
import texar.torch as tx
from forte.models.da_rl import MetaModule
from texar.torch.modules.networks.networks import FeedForwardNetwork


class TestMetaModule(unittest.TestCase):

    def setUp(self):
        hparams = {
            "layers": [
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 32,
                        "out_features": 64
                    }
                },
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 64,
                        "out_features": 128
                    }
                }
            ]
        }

        self.nn_module = FeedForwardNetwork(hparams=hparams)

        print("hparam[layer] *2 = ", len(hparams["layers"]) * 2)
        print("nn trainable_variables = ", self.nn_module.trainable_variables)

        self.nn_magic_model = MetaModule(self.nn_module)

    def test_update_params(self):
        old_num_param = len(self.nn_magic_model._buffers)
        old_num_module_param = len(self.nn_magic_model._modules)
        print("old_num_param, old_num_param_module = ", old_num_param, old_num_module_param)

        grads = {name: torch.zeros_like(param)
                 for name, param in self.nn_module.named_parameters()}
        self.nn_magic_model.update_params(grads)

        new_num_param = len(self.nn_magic_model._buffers)
        new_num_module_param = len(self.nn_magic_model._modules)

        print("new_num_param, new_num_param_module = ", new_num_param, new_num_module_param)

        self.assertEqual(old_num_module_param, new_num_module_param)
        self.assertEqual(old_num_param, new_num_param)

    def test_forward_with_nn_module(self):
        outputs = self.nn_magic_model(torch.ones(64, 16, 32))
        print("output size = ", outputs)
        print("expected nn output_size = ", self.nn_module.output_size)
        self.assertEqual(outputs.size(-1), self.nn_module.output_size)

    def test_forward_with_texar_bert(self):
        pretrained_model_name = 'bert-base-uncased'
        config_classifier = {
            "name": "bert_classifier",
            "hidden_size": 768,
            "clas_strategy": "cls_time",
            "dropout": 0.1,
            "num_classes": 2
        }
        bert_model = tx.modules.BERTClassifier(
            pretrained_model_name=pretrained_model_name,
            hparams=config_classifier)
        bert_magic_model = MetaModule(bert_model)

        input_ids = torch.ones(2, 16, 1)
        segment_ids = torch.ones(2, 16, 1)
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)
        print("input_length: ", input_length)
        print("input_length shape: ", input_length.size())
        logits, _ = bert_magic_model(input_ids, input_length, segment_ids)
        print("logits shape: ", logits.size())
        # todo: assert?


    def test_forward_with_transformer_bert(self):
        pass
