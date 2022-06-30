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
from forte.data.modality import Modality
from forte.data.ontology.core import Grid
from ft.onto.base_ontology import ImagePayload
import numpy as np

from numpy import array_equal
from forte.data.ontology.top import Box, ImageAnnotation

from forte.data.data_pack import DataPack
import unittest


class ImageAnnotationTest(unittest.TestCase):
    """
    Test ImageAnnotation related ontologies like Edge and BoundingBox.
    """

    def setUp(self):
        self.datapack1 = DataPack("image")
        self.line = np.zeros((6, 12))
        self.line[2, 2] = 1
        self.line[3, 3] = 1
        self.line[4, 4] = 1
        ip = ImagePayload(self.datapack1, 0)
        ip.set_cache(self.line)

        self.datapack2 = DataPack("image")
        empty_img = np.zeros((641, 481))
        ip = ImagePayload(self.datapack2, 0)
        ip.set_cache(empty_img)

    def test_image_annotation(self):
        self.line = np.zeros((6, 12))
        self.line[2, 2] = 1
        self.line[3, 3] = 1
        self.line[4, 4] = 1
        ip = ImagePayload(self.datapack1, 0)
        ip.set_cache(self.line)
        ImageAnnotation(self.datapack1)
        self.assertEqual(
            self.datapack1.get_single(ImageAnnotation).image_payload_idx, 0
        )

        self.assertTrue(
            array_equal(
                self.datapack1.get_payload_at(Modality.Image, 0).cache,
                self.line,
            )
        )
        new_pack = DataPack.from_string(self.datapack1.to_string())
        self.assertEqual(
            new_pack.audio_annotations, self.datapack1.audio_annotations
        )

    def test_wrong_box(self):
        empty_img = np.zeros((641, 481))
        ip = ImagePayload(self.datapack2, 0)
        ip.set_cache(empty_img)
        # test a box has a wrong shape (negative height)
        def fn():
            Box(self.datapack2, -1, 1)

        self.assertRaises(ValueError, fn)

        # test a box has a wrong center (negative center y coordinate)
        def fn():
            Box(self.datapack2, 1, 1, -1, 1)

        self.assertRaises(ValueError, fn)

        # test a box has a wrong center (not large enough to contain the box)
        def fn():
            Box(self.datapack2, 4, 4, 1, 1)

        self.assertRaises(ValueError, fn)

        # test a box that is too large
        def fn():
            Box(self.datapack2, 641, 481, 3, 3)

        self.assertRaises(ValueError, fn)

        # test a box that is out of the image
        def fn():
            Box(self.datapack2, 30, 30, 630, 470)

        self.assertRaises(ValueError, fn)

    def test_incomplete_box(self):
        # test a box doesn't have a center
        def fn():
            b = Box(self.datapack2, 1, 1)
            b.corners

        self.assertRaises(ValueError, fn)

    def test_simple_box(self):
        # test a box not associated with any grid
        # test its propertys are correct

        b = Box(self.datapack2, 640, 480, 320, 240)
        self.assertEqual(b.cx, 240)
        self.assertEqual(b.cy, 320)
        self.assertFalse(b.is_grid_associated)
        self.assertEqual(b.box_center, (320, 240))
        self.assertEqual(b.corners, ((0, 0), (0, 480), (640, 0), (640, 480)))
        self.assertEqual(b.box_min_y, 0)
        self.assertEqual(b.box_min_x, 0)
        self.assertEqual(b.box_max_y, 640)
        self.assertEqual(b.box_max_x, 480)
        self.assertEqual(b.area, 640 * 480)

    def test_ground_truth_box(self):
        # test a box shape and absolute position is given
        # we can compute the relative position to a grid cell
        empty_img = np.zeros((641, 481))
        ip = ImagePayload(self.datapack2, 0)
        ip.set_cache(empty_img)
        b = Box(self.datapack2, 640, 480, 320, 240)
        g = Grid(64, 48, 640, 480)  # each grid cell has a shape of 10 x 10
        # to simplify the test, we use the Box has the same shape as the image
        self.assertFalse(b.is_grid_associated)
        b.set_grid_cell_center(g, 1, 1)
        self.assertTrue(b.is_grid_associated)
        self.assertEqual(b.grid_cy, 15)
        self.assertEqual(b.grid_cx, 15)
        self.assertEqual(b.grid_cell_center, (15, 15))
        # 320 - 15 = 305, 240 - 15 = 225
        self.assertEqual(b.cy_offset, 305)
        self.assertEqual(b.cx_offset, 225)

    def test_predicted_box(self):
        # test a box has a predicted shape and offset given a grid cell
        # we can compute the absolute position
        empty_img = np.zeros((641, 481))
        ip = ImagePayload(self.datapack2, 0)
        ip.set_cache(empty_img)
        b = Box(self.datapack2, 640, 480)
        g = Grid(64, 48, 640, 480)  # each grid cell has a shape of 10 x 10
        # to simplify the test, we use the Box has the same shape as the image
        self.assertFalse(b.is_grid_associated)
        b.set_grid_cell_center(g, 1, 1)
        self.assertTrue(b.is_grid_associated)
        # suppose we have a predicted offset (305, 225)
        b.set_offset(305, 225)
        self.assertEqual(b.offset, (305, 225))
        self.assertEqual(b.cy_offset, 305)
        self.assertEqual(b.cx_offset, 225)
        self.assertEqual(b.grid_cy, 15)
        self.assertEqual(b.grid_cx, 15)
        self.assertEqual(b.grid_cell_center, (15, 15))
        # 305 + 15 = 320, 225 + 15 = 240
        self.assertEqual(b.cy, 320)
        self.assertEqual(b.cx, 240)
