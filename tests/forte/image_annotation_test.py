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
import os
import unittest
import numpy as np
from typing import Dict

from numpy import array_equal
from forte.data.ontology.top import ImageAnnotation
from forte.data.data_pack import DataPack


class ImageAnnotationTest(unittest.TestCase):
    """
    Test ImageAnnotation related ontologies like Edge and BoundingBox.
    """

    def setUp(self):
        self.datapack = DataPack("image")
        self.line = np.zeros((6, 12))
        self.line[2, 2] = 1
        self.line[3, 3] = 1
        self.line[4, 4] = 1
        self.datapack.payloads.append(self.line)
        self.datapack.image_annotations.append(
            ImageAnnotation(self.datapack, 0)
        )

    def test_image_annotation(self):
        self.assertEqual(
            self.datapack.image_annotations[0].image_payload_idx, 0
        )

        self.assertTrue(
            array_equal(self.datapack.image_annotations[0].image, self.line)
        )
