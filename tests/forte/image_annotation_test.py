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
import numpy as np

from numpy import array_equal
from forte.data.ontology.top import Box, ImageAnnotation

from ft.onto.base_ontology import ImagePayload

from forte.data.data_pack import DataPack
import unittest


class ImageAnnotationTest(unittest.TestCase):
    """
    Test ImageAnnotation related ontologies like Edge and Box.
    """

    def setUp(self):
        self.datapack = DataPack("image")
        self.line = np.zeros((6, 12))
        self.line[2, 2] = 1
        self.line[3, 3] = 1
        self.line[4, 4] = 1
        ip = ImagePayload(self.datapack, 0)
        ip.set_cache(self.line)
        ImageAnnotation(self.datapack)

    def test_image_annotation(self):
        self.assertEqual(
            self.datapack.get_single(ImageAnnotation).image_payload_idx, 0
        )

        self.assertTrue(
            array_equal(
                self.datapack.get_payload_at(Modality.Image, 0).cache, self.line
            )
        )
        new_pack = DataPack.from_string(self.datapack.to_string())
        self.assertEqual(
            new_pack.audio_annotations, self.datapack.audio_annotations
        )

    def testBox(self):

        b1 = Box(self.datapack, [2, 2], [7, 7], 0)
        self.assertEqual(b1.corners, [(2, 2), (2, 7), (7, 2), (7, 7)])
        self.assertEqual(b1.area, 25)
        self.assertEqual(b1.center, (4, 4))

        b2 = Box.init_from_center_n_shape(self.datapack, 4, 4, 5, 5)
        self.assertEqual(b2.corners, [(2, 2), (2, 7), (7, 2), (7, 7)])
        self.assertEqual(b2.area, 25)
        self.assertEqual(b2.center, (4, 4))

        def wrong_box():
            # negative x coordinate
            Box(self.datapack, [-2, 2], [7, 7], 0)

        self.assertRaises(ValueError, wrong_box)

        def wrong_box():
            # negative y coordinate
            Box(self.datapack, [2, -2], [7, 7], 0)

        self.assertRaises(ValueError, wrong_box)

        def wrong_box():
            # negative width
            Box(self.datapack, [3, 2], [2, 7], 0)

        self.assertRaises(ValueError, wrong_box)

        def wrong_box():
            # negative height
            Box(self.datapack, [2, 2], [2, 1], 0)

        self.assertRaises(ValueError, wrong_box)

        b3 = Box.init_from_center_n_shape(self.datapack, 4, 4, 5, 5)
