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

from typing import List

from forte.data.extractor.feature import Feature


class FeatureTest(unittest.TestCase):
    def setUp(self):
        data1: List = [7, 8, 9]
        pad_id1: int = 0
        dim1: int = 1
        self.feature1: Feature = Feature(data1, pad_id1, dim1)

        data2: List = [[6, 11, 2], [7, 8], [6, 7, 5, 4]]
        pad_id2: int = 0
        dim2: int = 2
        self.feature2: Feature = Feature(data2, pad_id2, dim2)

        data3: List = [[[0, 1, 0], [1, 0, 0], [1, 0, 0]],
                       [[1, 0, 0], [0, 1, 0]]]
        pad_id3: List = [0, 0, 1]
        dim3: int = 2
        self.feature3: Feature = Feature(data3, pad_id3, dim3)

    def test_is_base_feature(self):
        self.assertTrue(self.feature1.is_base_feature())
        self.assertFalse(self.feature2.is_base_feature())
        self.assertFalse(self.feature3.is_base_feature())

    def test_get_sub_feature(self):
        sub_features2: List[Feature] = self.feature2.get_sub_features()
        for sub_feature in sub_features2:
            self.assertTrue(sub_feature.is_base_feature())
        self.assertEqual(sub_features2[0].data, [6, 11, 2])
        self.assertEqual(sub_features2[1].data, [7, 8])
        self.assertEqual(sub_features2[2].data, [6, 7, 5, 4])

        sub_features3: List[Feature] = self.feature3.get_sub_features()
        for sub_feature in sub_features3:
            self.assertTrue(sub_feature.is_base_feature())
        self.assertEqual(sub_features3[0].data,
                         [[0, 1, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(sub_features3[1].data, [[1, 0, 0], [0, 1, 0]])

    def test_get_len(self):
        self.assertEqual(self.feature1.get_len(), 3)
        self.assertEqual(self.feature2.get_len(), 3)
        self.assertEqual(self.feature3.get_len(), 2)

    def test_pad(self):
        pass

    def test_unroll(self):
        pass


if __name__ == '__main__':
    unittest.main()
