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
# pylint: disable=protected-access
"""
Unit tests for MetaAugmentationWrapper.
"""
import unittest
import torch
import texar.torch as tx
from torch import optim
from transformers import BertForMaskedLM

from forte.models.da_rl.aug_wrapper import MetaAugmentationWrapper
from forte.models.da_rl.aug_wrapper import (
    _texar_bert_adam_delta,
    _torch_adam_delta,
)


class TestMetaAugmentationWrapper(unittest.TestCase):
    def setUp(self):
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

        self.lr = 4e-5
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optim = tx.core.BertAdam(
            self.optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-6,
            lr=self.lr,
        )

        mask_token_id = 100
        self.num_aug = 3

        self.wrapper = MetaAugmentationWrapper(
            self.model,
            self.optim,
            mask_token_id,
            torch.device("cpu"),
            self.num_aug,
        )

        self.seq_length = 16
        self.batch_size = 2
        self.num_bert_token = 30522

    def test_augment_instance(self):
        input_ids = torch.randint(10, 90, (self.seq_length,))
        label_ids = input_ids
        input_mask = torch.ones_like(input_ids)
        segment_ids = torch.zeros_like(input_ids)
        features = (input_ids, input_mask, segment_ids, label_ids)

        self.wrapper.reset_model()
        augmented_features = self.wrapper.augment_instance(features)

        (
            aug_probs,
            input_mask_aug,
            segment_ids_aug,
            label_ids_aug,
        ) = augmented_features

        # aug_probs is soft embedding
        self.assertEqual(
            aug_probs.size(),
            torch.Size([self.num_aug, self.seq_length, self.num_bert_token]),
        )
        self.assertEqual(
            input_mask_aug.size(), torch.Size([self.num_aug, self.seq_length])
        )
        self.assertEqual(
            segment_ids_aug.size(), torch.Size([self.num_aug, self.seq_length])
        )
        self.assertEqual(
            label_ids_aug.size(), torch.Size([self.num_aug, self.seq_length])
        )
        for label_id in label_ids_aug:
            self.assertTrue(torch.equal(label_id, label_ids))

    def test_augment_batch(self):
        batch_input_id = torch.randint(
            10, 90, (self.batch_size, self.seq_length)
        )
        batch_label_id = batch_input_id
        input_mask = torch.ones_like(batch_input_id)
        segment_ids = torch.zeros_like(batch_input_id)
        features = (batch_input_id, input_mask, segment_ids, batch_label_id)

        augmented_features = self.wrapper.augment_batch(features)

        (
            aug_probs,
            input_mask_aug,
            segment_ids_aug,
            label_ids_aug,
        ) = augmented_features

        # aug_probs is soft embedding
        self.assertEqual(
            aug_probs.size(),
            torch.Size(
                [2 * self.batch_size, self.seq_length, self.num_bert_token]
            ),
        )
        self.assertEqual(
            input_mask_aug.size(),
            torch.Size([2 * self.batch_size, self.seq_length]),
        )
        self.assertEqual(
            segment_ids_aug.size(),
            torch.Size([2 * self.batch_size, self.seq_length]),
        )
        self.assertEqual(
            label_ids_aug.size(),
            torch.Size([2 * self.batch_size, self.seq_length]),
        )
        for i in range(2):
            original_or_aug_label_ids = label_ids_aug[
                i * self.batch_size : (i + 1) * self.batch_size
            ]
            self.assertTrue(
                torch.equal(original_or_aug_label_ids, batch_label_id)
            )

    def test_eval_batch(self):
        batch_input_id = torch.randint(
            10, 90, (self.batch_size, self.seq_length)
        )
        batch_label_id = batch_input_id
        input_mask = torch.ones_like(batch_input_id)
        segment_ids = torch.zeros_like(batch_input_id)
        features = (batch_input_id, input_mask, segment_ids, batch_label_id)

        batch_loss = self.wrapper.eval_batch(features)
        self.assertFalse(
            torch.equal(
                batch_loss,
                torch.Tensor(
                    [
                        0.0,
                    ]
                ),
            )
        )

    def test_bert_adam_delta(self):
        grads = {
            param: torch.ones_like(param)
            for name, param in self.model.named_parameters()
        }

        deltas = _texar_bert_adam_delta(grads, self.model, self.optim)

        num_params = 0
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(deltas[name], torch.zeros_like(param)))
            num_params += 1
        self.assertEqual(len(deltas), num_params)

    def test_adam_delta(self):
        grads = {
            param: torch.ones_like(param)
            for name, param in self.model.named_parameters()
        }

        adam_optim = optim.Adam(self.optimizer_grouped_parameters, lr=self.lr)

        deltas = _torch_adam_delta(grads, self.model, adam_optim)

        num_params = 0
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(deltas[name], torch.zeros_like(param)))
            num_params += 1
        self.assertEqual(len(deltas), num_params)
