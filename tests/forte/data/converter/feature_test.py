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
from typing import List, Union, Type
import torch

from forte.data.converter import Feature


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.feature1: Feature = self.create_feature1()
        self.feature2: Feature = self.create_feature2()
        self.feature3: Feature = self.create_feature3()

    def test_is_leaf_feature(self):
        self.assertTrue(self.feature1.leaf_feature)
        self.assertFalse(self.feature2.leaf_feature)
        self.assertFalse(self.feature3.leaf_feature)

    def test_get_sub_feature(self):
        sub_features2: List[Feature] = self.feature2.sub_features
        for sub_feature in sub_features2:
            self.assertTrue(sub_feature.leaf_feature)
        self.assertEqual(sub_features2[0]._data, [6, 11, 2])
        self.assertEqual(sub_features2[1]._data, [7, 8])
        self.assertEqual(sub_features2[2]._data, [6, 7, 5, 4])

        sub_features3: List[Feature] = self.feature3.sub_features
        for sub_feature in sub_features3:
            self.assertTrue(sub_feature.leaf_feature)
        self.assertEqual(
            sub_features3[0]._data, [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        )
        self.assertEqual(sub_features3[1]._data, [[1, 0, 0], [0, 1, 0]])

    def test_get_len(self):
        self.assertEqual(len(self.feature1), 3)
        self.assertEqual(len(self.feature2), 3)
        self.assertEqual(len(self.feature3), 2)

    def test_pad(self):
        self.feature1.pad(4)
        self.assertEqual(self.feature1._data, [7, 8, 9, 0])

        self.feature1 = self.create_feature1()
        self.feature1.pad(6)
        self.assertEqual(self.feature1._data, [7, 8, 9, 0, 0, 0])

        self.feature1 = self.create_feature1(
            data=[[1, 0, 0], [0, 1, 0]], pad_id=[0, 0, 1], dim=1
        )
        self.feature1.pad(4)
        self.assertEqual(
            self.feature1._data, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        )

        self.feature2.pad(4)
        self.assertEqual(len(self.feature2), 4)
        leaf_feature_data = [i._data for i in self.feature2.sub_features]
        self.assertEqual(
            leaf_feature_data[:-1], [[6, 11, 2], [7, 8], [6, 7, 5, 4]]
        )
        self.assertEqual(len(leaf_feature_data[-1]), 0)

        self.feature3.pad(4)
        self.assertEqual(len(self.feature3), 4)
        leaf_feature_data = [i._data for i in self.feature3.sub_features]
        self.assertEqual(
            leaf_feature_data[:-2],
            [[[0, 1, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]],
        )
        self.assertEqual(len(leaf_feature_data[-2]), 0)
        self.assertEqual(len(leaf_feature_data[-1]), 0)

    def test_unroll(self):
        self.feature1.pad(4)
        feature, mask = self.feature1.data
        self.assertEqual(feature, [7, 8, 9, 0])
        self.assertEqual(mask, [[1, 1, 1, 0]])

        self.feature2.pad(4)
        for sub_feature in self.feature2.sub_features:
            sub_feature.pad(4)
        feature, mask = self.feature2.data
        self.assertEqual(
            feature, [[6, 11, 2, 0], [7, 8, 0, 0], [6, 7, 5, 4], [0, 0, 0, 0]]
        )
        self.assertEqual(
            mask,
            [
                [1, 1, 1, 0],
                [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            ],
        )

        self.feature3.pad(4)
        for sub_feature in self.feature3.sub_features:
            sub_feature.pad(3)
        feature, mask = self.feature3.data
        self.assertEqual(
            feature,
            [
                [[0, 1, 0], [1, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ],
        )
        self.assertEqual(
            mask, [[1, 1, 0, 0], [[1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0]]]
        )

    def test_unroll_nopad(self):
        feature, mask = self.feature1.data
        self.assertEqual(feature, [7, 8, 9])

        feature, mask = self.feature2.data
        self.assertEqual(feature, [[6, 11, 2], [7, 8], [6, 7, 5, 4]])

        feature, mask = self.feature3.data
        self.assertEqual(
            feature, [[[0, 1, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]
        )

    def test_unroll_dtype_float(self):
        self.feature1 = self.create_feature1(dtype=torch.float)
        self.assertEqual(self.feature1.dtype, torch.float)

        self.feature2 = self.create_feature2(dtype=torch.float)
        self.assertEqual(self.feature2.dtype, torch.float)

        for sub_feature in self.feature2.sub_features:
            self.assertEqual(sub_feature.dtype, torch.float)

    def test_unroll_dtype_str(self):
        self.feature1 = self.create_feature1(
            data=["I", "like", "Forte"], pad_id="<PAD>", dtype=str
        )
        self.assertEqual(self.feature1.dtype, str)
        self.assertEqual(self.feature1.data[0], ["I", "like", "Forte"])

        self.feature1.pad(max_len=4)
        self.assertEqual(self.feature1.data[0], ["I", "like", "Forte", "<PAD>"])

        self.feature2 = self.create_feature2(
            data=[["I"], ["l", "i", "k", "e"], ["F", "o", "r", "t", "e"]],
            pad_id="<PAD>",
            dtype=str,
        )
        for sub_feature in self.feature2.sub_features:
            sub_feature.pad(max_len=5)

        self.assertEqual(self.feature2.dtype, str)
        self.assertEqual(
            self.feature2.data[0],
            [
                ["I", "<PAD>", "<PAD>", "<PAD>", "<PAD>"],
                ["l", "i", "k", "e", "<PAD>"],
                ["F", "o", "r", "t", "e"],
            ],
        )

    def create_feature1(
        self, data=None, pad_id=None, dim=None, need_pad=True, dtype=None
    ):
        data: List = [7, 8, 9] if data is None else data
        pad_id: int = 0 if pad_id is None else pad_id
        dim: int = 1 if dim is None else dim
        dtype = torch.long if dtype is None else dtype
        feature: Feature = Feature(
            data,
            {
                "pad_value": pad_id,
                "dim": dim,
                "dtype": dtype,
                "need_pad": need_pad,
            },
        )

        return feature

    def create_feature2(
        self, data=None, pad_id=None, dim=None, need_pad=True, dtype=None
    ):
        data: List = (
            [[6, 11, 2], [7, 8], [6, 7, 5, 4]] if data is None else data
        )
        pad_id: int = 0 if pad_id is None else pad_id
        dim: int = 2 if dim is None else dim
        dtype = torch.long if dtype is None else dtype
        feature: Feature = Feature(
            data,
            {
                "pad_value": pad_id,
                "dim": dim,
                "dtype": dtype,
                "need_pad": need_pad,
            },
        )

        return feature

    def create_feature3(
        self, data=None, pad_id=None, dim=None, need_pad=True, dtype=torch.long
    ):
        data: List = (
            [[[0, 1, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]
            if data is None
            else data
        )
        pad_id: List = [0, 0, 1] if pad_id is None else pad_id
        dim: int = 2 if dim is None else dim
        feature: Feature = Feature(
            data,
            {
                "pad_value": pad_id,
                "dim": dim,
                "dtype": dtype,
                "need_pad": need_pad,
            },
        )

        return feature


if __name__ == "__main__":
    unittest.main()
