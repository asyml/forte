# Copyright 2022 The Forte Authors. All Rights Reserved.
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
Unit tests for Grids.
"""
import unittest
import numpy as np

from numpy import array_equal
from forte.data.ontology.top import Grids
from forte.data.data_pack import DataPack
from forte.data.ontology.top import ImageAnnotation


class GridsTest(unittest.TestCase):
    """
    Test Grids related ontologies and operations.
    """

    def setUp(self):
        self.datapack = DataPack("image")
        line = np.zeros((6, 12))
        line[2, 2] = 1
        line[3, 3] = 1
        line[4, 4] = 1
        self.datapack.payloads.append(line)
        self.datapack.image_annotations.append(
            ImageAnnotation(self.datapack, 0)
        )

        grids = Grids(self.datapack, 3, 4)

        self.datapack.grids.append(grids)
        self.zeros = np.zeros((6, 12))
        self.ref_arr = np.zeros((6, 12))
        self.ref_arr[2, 2] = 1
        self.datapack.payloads.append(self.ref_arr)

    def test_grids(self):

        self.assertTrue(
            array_equal(self.datapack.grids[0].get_grid_cell(0, 0), self.zeros)
        )

        self.assertTrue(
            array_equal(
                self.datapack.grids[0].get_grid_cell(1, 0), self.ref_arr
            )
        )

    def test_get_grid_cell_value_error(self):
        def fn1():
            self.datapack.grids[0].get_grid_cell(3, 0)

        self.assertRaises(ValueError, fn1)

        def fn2():
            self.datapack.grids[0].get_grid_cell(0, 4)

        self.assertRaises(ValueError, fn2)

        def fn3():
            self.datapack.grids[0].get_grid_cell(-1, 0)

        self.assertRaises(ValueError, fn3)

        def fn4():
            self.datapack.grids[0].get_grid_cell(0, -1)

        self.assertRaises(ValueError, fn4)
