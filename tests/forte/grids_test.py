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
Unit tests for ImageAnnotation.
"""
import unittest
import numpy as np

from numpy import array_equal
from forte.data.ontology.top import Grids
from forte.data.data_pack import DataPack
from forte.data.ontology.top import (
    ImageAnnotation
)


class ImageAnnotationTest(unittest.TestCase):
    """
    Test ImageAnnotation related ontologies like Edge and BoundingBox.
    """

    def setUp(self):
        self.datapack = DataPack("image")
        line = np.zeros((6, 12))
        line[2,2] = 1
        line[3,3] = 1
        line[4,4] = 1
        self.datapack.payloads.append(line)
        self.datapack.image_annotations.add(ImageAnnotation(self.datapack, 0))
        
        grids = Grids(self.datapack, (3,4))

        self.datapack.grids.add(grids)

    def test_image_annotation(self):
        ref_arr = np.zeros((2,3))
        ref_arr[0,2] = 1
        self.assertTrue(
            array_equal(self.datapack.grids[0].get_grid_cell(1,0,0), ref_arr)
        )

        ref_arr2 = np.zeros((2,3))
        ref_arr2[1, 0] = 1
        self.assertTrue(
            array_equal(self.datapack.grids[0].get_grid_cell(1,1,0), ref_arr2)
        )