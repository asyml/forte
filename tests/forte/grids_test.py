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
Unit tests for Grid.
"""

import unittest
from forte.data.ontology.core import Grid
import numpy as np

from numpy import array_equal

from forte.data.data_pack import DataPack
from forte.data.ontology.top import ImageAnnotation


class GridTest(unittest.TestCase):
    """
    Test Grid related ontologies and operations.
    """

    def setUp(self):
        self.datapack = DataPack("image")

        self.line = np.zeros((4, 6))
        self.line[1, 1] = 1
        self.line[2, 2] = 1
        self.line[3, 3] = 1
        # self.line[4, 4] = 1
        self.datapack.payloads.append(self.line)
        self.datapack.image_annotations.append(
            ImageAnnotation(self.datapack, 0)
        )

        image_height, image_width = self.line.shape
        grid = Grid(2, 3, image_height, image_width)

        self.grid = grid
        self.zeros = np.zeros((4, 6))
        self.ref_arr = np.zeros((4, 6))
        self.ref_arr[2, 2] = 1
        self.ref_arr[3, 3] = 1
        self.datapack.payloads.append(self.ref_arr)

    def test_grids(self):

        self.assertTrue(
            array_equal(self.grid.get_grid_cell(self.line, 0, 1), self.zeros)
        )

        self.assertTrue(
            array_equal(self.grid.get_grid_cell(self.line, 1, 1), self.ref_arr)
        )

        self.grid.get_grid_cell_center(1, 1)

    def test_get_grid_cell_value_error(self):
        def fn1():
            self.grid.get_grid_cell(self.line, 2, 0)

        self.assertRaises(ValueError, fn1)

        def fn2():
            self.grid.get_grid_cell(self.line, 0, 3)

        self.assertRaises(ValueError, fn2)

        def fn3():
            self.grid.get_grid_cell(self.line, -1, 0)

        self.assertRaises(ValueError, fn3)

        def fn4():
            self.grid.get_grid_cell(self.line, 0, -1)

        self.assertRaises(ValueError, fn4)
