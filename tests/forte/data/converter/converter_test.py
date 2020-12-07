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

from typing import List

import torch

from forte.data.converter.converter import Converter
from forte.data.converter.feature import Feature


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.converter: Converter = Converter()

    def test_convert1(self):
        features1: List[Feature] = self.create_features1()

        tensor, mask_list = self.converter.convert(features1)
        self.assertTrue(
            torch.allclose(tensor,
                           torch.tensor(
                               [[7, 8, 9, 0], [1, 2, 5, 6], [4, 0, 0, 0]])))
        self.assertEqual(len(mask_list), 1)
        self.assertTrue(
            torch.equal(mask_list[0],
                        torch.tensor(
                            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                            dtype=torch.bool)))

    def test_convert2(self):
        features2: List[Feature] = self.create_features2()

        tensor, mask_list = self.converter.convert(features2)
        self.assertTrue(
            torch.allclose(tensor,
                           torch.tensor(
                               [[[6, 11, 2, 0],
                                 [7, 8, 0, 0],
                                 [6, 7, 5, 4],
                                 [0, 0, 0, 0]],
                                [[4, 3, 0, 0],
                                 [7, 6, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[6, 3, 0, 0],
                                 [5, 0, 0, 0],
                                 [7, 12, 0, 0],
                                 [7, 11, 0, 0]]])))
        self.assertEqual(len(mask_list), 2)
        self.assertTrue(
            torch.equal(mask_list[0],
                        torch.tensor(
                            [[1, 1, 1, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 1]],
                            dtype=torch.bool
                        )))
        self.assertTrue(
            torch.equal(mask_list[1],
                        torch.tensor(
                            [[[1, 1, 1, 0],
                              [1, 1, 0, 0],
                              [1, 1, 1, 1],
                              [0, 0, 0, 0]],
                             [[1, 1, 0, 0],
                              [1, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]],
                             [[1, 1, 0, 0],
                              [1, 0, 0, 0],
                              [1, 1, 0, 0],
                              [1, 1, 0, 0]]],
                            dtype=torch.bool)))

    def test_convert3(self):
        features3: List[Feature] = self.create_features3()

        tensor, mask_list = self.converter.convert(features3)
        self.assertTrue(
            torch.allclose(tensor,
                           torch.tensor(
                               [[[[0, 1, 0], [1, 0, 0]],
                                 [[1, 0, 0], [0, 1, 0]],
                                 [[0, 0, 1], [0, 0, 1]]],
                                [[[1, 0, 0], [0, 1, 0]],
                                 [[0, 0, 1], [0, 0, 1]],
                                 [[0, 0, 1], [0, 0, 1]]],
                                [[[0, 1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [1, 0, 0]],
                                 [[0, 1, 0], [1, 0, 0]]]
                                ])))
        self.assertEqual(len(mask_list), 2)
        self.assertTrue(
            torch.equal(mask_list[0],
                        torch.tensor(
                            [[1, 1, 0],
                             [1, 0, 0],
                             [1, 1, 1]],
                            dtype=torch.bool
                        )))
        self.assertTrue(
            torch.equal(mask_list[1],
                        torch.tensor(
                            [[[1, 1],
                              [1, 1],
                              [0, 0]],
                             [[1, 1],
                              [0, 0],
                              [0, 0]],
                             [[1, 0],
                              [1, 1],
                              [1, 1]]],
                            dtype=torch.bool)))

    def test_convert_dtype(self):
        features1: List[Feature] = self.create_features1(dtype=torch.float)
        tensor, mask_list = self.converter.convert(features1)
        self.assertTrue(
            torch.allclose(tensor,
                           torch.tensor(
                               [[7, 8, 9, 0], [1, 2, 5, 6], [4, 0, 0, 0]],
                               dtype=torch.float)))

        features2: List[Feature] = self.create_features2(dtype=torch.float)
        tensor, mask_list = self.converter.convert(features2)
        self.assertTrue(
            torch.allclose(tensor,
                           torch.tensor(
                               [[[6, 11, 2, 0],
                                 [7, 8, 0, 0],
                                 [6, 7, 5, 4],
                                 [0, 0, 0, 0]],
                                [[4, 3, 0, 0],
                                 [7, 6, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[6, 3, 0, 0],
                                 [5, 0, 0, 0],
                                 [7, 12, 0, 0],
                                 [7, 11, 0, 0]]],
                               dtype=torch.float)))

    def create_features1(self,
                         data_list=None,
                         pad_id=None,
                         dim=None,
                         dtype=torch.long):
        if not data_list:
            data_list = [[7, 8, 9], [1, 2, 5, 6], [4]]
            pad_id: int = 0 if pad_id is None else pad_id
            dim: int = 1 if dim is None else dim

        features: List[Feature] = []
        for data in data_list:
            feature: Feature = Feature(data, {"pad_value": pad_id,
                                              "dim": dim,
                                              "dtype": dtype
                                             })
            features.append(feature)

        return features

    def create_features2(self,
                         data_list=None,
                         pad_id=None,
                         dim=None,
                         dtype=torch.long):
        if not data_list:
            data_list = [[[6, 11, 2], [7, 8], [6, 7, 5, 4]],
                         [[4, 3], [7, 6]],
                         [[6, 3], [5], [7, 12], [7, 11]]]
            pad_id: int = 0 if pad_id is None else pad_id
            dim: int = 2 if dim is None else dim

        features: List[Feature] = []
        for data in data_list:
            feature: Feature = Feature(data, {"pad_value": pad_id,
                                              "dim": dim,
                                              "dtype": dtype
                                             })
            features.append(feature)

        return features

    def create_features3(self,
                         data_list=None,
                         pad_id=None,
                         dim=None,
                         dtype=torch.long):
        if not data_list:
            data_list: List = \
                [  # Instance 1:
                    [[[0, 1, 0], [1, 0, 0]],
                     [[1, 0, 0], [0, 1, 0]]],
                    # Instance 2:
                    [[[1, 0, 0], [0, 1, 0]]],
                    # Instance 3:
                    [[[0, 1, 0]],
                     [[1, 0, 0], [1, 0, 0]],
                     [[0, 1, 0], [1, 0, 0]]]
                ]
            pad_id: List = [0, 0, 1] if pad_id is None else pad_id
            dim: int = 2 if dim is None else dim

        features: List[Feature] = []
        for data in data_list:
            feature: Feature = Feature(data, {"pad_value": pad_id,
                                              "dim": dim,
                                              "dtype": dtype
                                             })
            features.append(feature)

        return features


if __name__ == '__main__':
    unittest.main()
