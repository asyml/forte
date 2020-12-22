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
import os
import numpy as np
from typing import List
import unittest
import torch

from forte.data.converter import Converter
from forte.data.converter import Feature


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.converter: Converter = Converter({})

    def test_convert1(self):
        features1: List[Feature] = self.create_features1()

        data, masks_list = self.converter.convert(features1)
        self.assertTrue(
            torch.allclose(data,
                           torch.tensor(
                               [[7, 8, 9, 0], [1, 2, 5, 6], [4, 0, 0, 0]])))
        self.assertEqual(len(masks_list), 1)
        self.assertTrue(
            torch.equal(masks_list[0],
                        torch.tensor(
                            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
                            dtype=torch.bool)))

    def test_convert2(self):
        features2: List[Feature] = self.create_features2()

        data, masks_list = self.converter.convert(features2)
        self.assertTrue(
            torch.allclose(data,
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
                                 [7, 11, 0, 0]]], dtype=torch.long)))
        self.assertEqual(len(masks_list), 2)
        self.assertTrue(
            torch.equal(masks_list[0],
                        torch.tensor(
                            [[1, 1, 1, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 1]],
                            dtype=torch.bool)))
        self.assertTrue(
            torch.equal(masks_list[1],
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

        data, masks_list = self.converter.convert(features3)
        self.assertTrue(
            torch.allclose(data,
                           torch.tensor(
                               [[[[0, 1, 0], [1, 0, 0]],
                                 [[1, 0, 0], [0, 1, 0]],
                                 [[0, 0, 1], [0, 0, 1]]],
                                [[[1, 0, 0], [0, 1, 0]],
                                 [[0, 0, 1], [0, 0, 1]],
                                 [[0, 0, 1], [0, 0, 1]]],
                                [[[0, 1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [1, 0, 0]],
                                 [[0, 1, 0], [1, 0, 0]]]], dtype=torch.long)))
        self.assertEqual(len(masks_list), 2)
        self.assertTrue(
            torch.equal(masks_list[0],
                        torch.tensor(
                            [[1, 1, 0],
                             [1, 0, 0],
                             [1, 1, 1]],
                            dtype=torch.bool)))
        self.assertTrue(
            torch.equal(masks_list[1],
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

    def test_convert_float(self):
        features1: List[Feature] = self.create_features1(dtype=torch.float)
        data, masks_list = self.converter.convert(features1)
        self.assertTrue(
            torch.allclose(data,
                           torch.tensor(
                               [[7, 8, 9, 0], [1, 2, 5, 6], [4, 0, 0, 0]],
                               dtype=torch.float)))

        features2: List[Feature] = self.create_features2(dtype=torch.float)
        data, masks_list = self.converter.convert(features2)
        self.assertTrue(
            torch.allclose(data,
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

    def test_convert_no_pad(self):
        features1: List[Feature] = self.create_features1(need_pad=False)

        converter: Converter = Converter({"to_numpy": False,
                                          "to_torch": False})
        data, _ = converter.convert(features1)
        self.assertTrue(
            np.array_equal(data,
                           [[7, 8, 9], [1, 2, 5, 6], [4]]))

    def test_convert_no_pad_but_to_torch(self):
        features1: List[Feature] = \
            self.create_features1(
                data_list=[[7], [1], [4]],
                need_pad=False)

        converter: Converter = Converter({})
        data, _ = converter.convert(features1)
        self.assertTrue(
            torch.allclose(data,
                           torch.tensor(
                               [[7], [1], [4]], dtype=torch.long)))

    def test_convert_no_to_torch(self):
        features1: List[Feature] = self.create_features1()

        converter: Converter = Converter({"to_torch": False})
        data, _ = converter.convert(features1)
        self.assertNotEqual(type(data), torch.Tensor)
        self.assertTrue(
            np.array_equal(data,
                           np.array([[7, 8, 9, 0], [1, 2, 5, 6], [4, 0, 0, 0]],
                                    dtype=np.long)))

    def test_state(self):
        converter_states = {"to_numpy": True, "to_torch": False}
        converter: Converter = Converter(converter_states)

        # Test state.
        self.assertEqual(converter.state, converter_states)

        # Test save & load state.
        tmp_state_file = ".tmp_converter_state"
        torch.save(converter.state, tmp_state_file)
        self.assertTrue(os.path.exists(tmp_state_file))

        recover_converter: Converter = Converter({})
        recover_converter.load_state(torch.load(tmp_state_file))

        self.assertEqual(recover_converter.state, converter_states)

        os.remove(tmp_state_file)
        self.assertFalse(os.path.exists(tmp_state_file))

    def create_features1(self,
                         data_list=None,
                         pad_id=0,
                         dim=1,
                         need_pad=True,
                         dtype=np.long):
        if not data_list:
            data_list = [[7, 8, 9], [1, 2, 5, 6], [4]]
            pad_id: int = 0 if pad_id is None else pad_id
            dim: int = 1 if dim is None else dim

        features: List[Feature] = []
        for data in data_list:
            feature: Feature = Feature(data, {"pad_value": pad_id,
                                              "dim": dim,
                                              "dtype": dtype,
                                              "need_pad": need_pad
                                              })
            features.append(feature)

        return features

    def create_features2(self,
                         data_list=None,
                         pad_id=None,
                         dim=None,
                         need_pad=True,
                         dtype=np.long):
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
                                              "dtype": dtype,
                                              "need_pad": need_pad
                                              })
            features.append(feature)

        return features

    def create_features3(self,
                         data_list=None,
                         pad_id=None,
                         dim=None,
                         need_pad=True,
                         dtype=np.long):
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
                                              "dtype": dtype,
                                              "need_pad": need_pad
                                              })
            features.append(feature)

        return features


if __name__ == '__main__':
    unittest.main()
