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
from texar.torch.modules.networks.networks import FeedForwardNetwork

from forte.models.da_rl import MetaModule, TexarBertMetaModule


class TestMetaModule(unittest.TestCase):
    def setUp(self):
        # nn config
        self.hparams = {
            "layers": [
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {"in_features": 32, "out_features": 64},
                },
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {"in_features": 64, "out_features": 128},
                },
            ]
        }
        # bert config
        self.pretrained_model_name = "bert-base-uncased"
        self.batch_size = 2
        self.seq_length = 16
        self.num_class = 3

    # helper function
    def recursive_module_param(self, model, buffer_param, module_param):
        len_buffer = len(model._buffers)
        buffer_param.append(len_buffer)
        len_sub_model = len(model._modules)
        module_param.append(len_sub_model)

        for _, sub_model in model._modules.items():
            self.recursive_module_param(sub_model, buffer_param, module_param)
        return buffer_param, module_param

    def test_update_params(self):
        nn_module = FeedForwardNetwork(hparams=self.hparams)
        nn_magic_model = MetaModule(nn_module)

        (
            old_num_buffer_param,
            old_num_module_param,
        ) = self.recursive_module_param(nn_magic_model, [], [])

        grads = {
            name: torch.zeros_like(param)
            for name, param in nn_module.named_parameters()
        }
        nn_magic_model.update_params(grads)

        (
            new_num_buffer_param,
            new_num_module_param,
        ) = self.recursive_module_param(nn_magic_model, [], [])

        self.assertEqual(old_num_module_param, new_num_module_param)
        self.assertEqual(old_num_buffer_param, new_num_buffer_param)

    def test_forward_with_nn_module(self):
        nn_module = FeedForwardNetwork(hparams=self.hparams)
        nn_magic_model = MetaModule(nn_module)

        outputs = nn_magic_model(torch.ones(64, 16, 32))
        self.assertEqual(outputs.size(-1), nn_module.output_size)

    def test_forward_with_texar_bert(self):
        config_classifier = {
            "name": "bert_classifier",
            "hidden_size": 768,
            "clas_strategy": "cls_time",
            "dropout": 0.1,
            "num_classes": self.num_class,
        }
        bert_model = tx.modules.BERTClassifier(
            pretrained_model_name=self.pretrained_model_name,
            hparams=config_classifier,
        )
        bert_magic_model = TexarBertMetaModule(bert_model)

        input_ids = torch.ones(
            (self.batch_size, self.seq_length), dtype=torch.long
        )
        segment_ids = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long
        )
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)

        logits, _ = bert_magic_model(input_ids, input_length, segment_ids)

        self.assertEqual(
            logits.size(), torch.Size([self.batch_size, self.num_class])
        )
