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
Unit tests for HTMLReader
"""
import tempfile
import unittest
import texar.torch as tx
import torch

from forte.processors.data_augment.algorithms.UDA import UDAIterator
from texar.torch.utils.shapes import get_rank


class UDAPipelineTest(unittest.TestCase):
    def setUp(self):
        self.pickle_data_dir = tempfile.TemporaryDirectory()
        max_seq_length = 128
        self.feature_types = {
            "input_ids": ["int64", "stacked_tensor", max_seq_length],
            "label_ids": ["int64", "stacked_tensor"]
        }

        self.unsup_feature_types = {
            "input_ids": ["int64", "stacked_tensor", max_seq_length],
            "aug_input_ids": ["int64", "stacked_tensor", max_seq_length],
            "label_ids": ["int64", "stacked_tensor"]
        }

        self.sample_feature = {
            "input_ids": [0] * 128,
            "label_ids": 0
        }

        self.unsup_sample_feature = {
            "input_ids": [0] * 128,
            "aug_input_ids": [1] * 128,
            "label_ids": 0
        }

        self.train_path = "{}/train.pkl".format(self.pickle_data_dir.name)
        self.test_path = "{}/predict.pkl".format(self.pickle_data_dir.name)
        self.unsup_path = "{}/unsupervised.pkl".format(self.pickle_data_dir.name)

        self.train_hparam = {
            "allow_smaller_final_batch": False,
            "batch_size": 5,
            "dataset": {
                "data_name": "data",
                "feature_types": self.feature_types,
                "files": self.train_path
            },
            "shuffle": True,
            "shuffle_buffer_size": None
        }

        self.test_hparam = {
            "allow_smaller_final_batch": True,
            "batch_size": 5,
            "dataset": {
                "data_name": "data",
                "feature_types": self.feature_types,
                "files": self.test_path
            },
            "shuffle": False
        }

        self.unsup_hparam = {
            "allow_smaller_final_batch": True,
            "batch_size": 5,
            "dataset": {
                "data_name": "data",
                "feature_types": self.unsup_feature_types,
                "files": self.unsup_path
            },
            "shuffle": True
        }

        self.output_sample_features_to_file(
            self.sample_feature,
            self.feature_types,
            self.train_path,
            dup_num=50
        )

        self.output_sample_features_to_file(
            self.sample_feature,
            self.feature_types,
            self.test_path,
            dup_num=10
        )

        self.output_sample_features_to_file(
            self.unsup_sample_feature,
            self.unsup_feature_types,
            self.unsup_path,
            dup_num=25
        )

    def tearDown(self):
        self.pickle_data_dir.cleanup()

    def output_sample_features_to_file(
            self,
            feature,
            feature_types,
            output_file,
            dup_num
    ):
        with tx.data.RecordData.writer(output_file, feature_types) as writer:
            for i in range(dup_num):
                writer.write(feature)

    def test_UDA_pipeline(self):
        train_dataset = tx.data.RecordData(
            hparams=self.train_hparam, device=torch.device("cpu"))
        test_dataset = tx.data.RecordData(
            hparams=self.test_hparam, device=torch.device("cpu"))
        unsup_dataset = tx.data.RecordData(
            hparams=self.unsup_hparam, device=torch.device("cpu"))
        sup_iterator = tx.data.DataIterator(
            {
                "train": train_dataset,
                "test": test_dataset,
            }
        )
        unsup_iterator = tx.data.DataIterator(
            {
                "unsup": unsup_dataset
            }
        )

        def unsup_forward_fn(batch):
            orig_input = batch["input_ids"]
            aug_input = batch["aug_input_ids"]

            orig_batch_size = orig_input.size(0)
            aug_batch_size = aug_input.size(0)
            num_category = 2
            orig_logits = torch.ones(orig_batch_size, num_category)
            aug_logits = torch.ones(aug_batch_size, num_category)
            return orig_logits, aug_logits

        iterator = UDAIterator(
            sup_iterator,
            unsup_iterator,
            unsup_forward_fn,
            softmax_temperature=1.0,
            confidence_threshold=-1,
            reduction="mean"
        )

        num_epoch = 10
        iterator.switch_to_dataset_unsup("unsup")

        for epoch in range(num_epoch):
            iterator.switch_to_dataset("train", use_unsup=True)

            for batch, unsup_batch, unsup_loss in iterator:
                self.assertLess(unsup_loss, 1e-5)

                sup_rank = get_rank(batch["input_ids"])
                self.assertEqual(sup_rank, 2)

                unsup_orig_rank = get_rank(unsup_batch["input_ids"])
                self.assertEqual(unsup_orig_rank, 2)

                unsup_aug_rank = get_rank(unsup_batch["aug_input_ids"])
                self.assertEqual(unsup_aug_rank, 2)

            iterator.switch_to_dataset("test", use_unsup=False)
            for batch, _, _ in iterator:
                sup_rank = get_rank(batch["input_ids"])
                self.assertEqual(sup_rank, 2)


if __name__ == "__main__":
    unittest.main()
