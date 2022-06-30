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
from forte.common.exception import ProcessExecutionException
from ft.onto.base_ontology import ImagePayload
from forte.data.data_pack import DataPack
import unittest
from sortedcontainers import SortedList
from forte.data.ontology.top import ImageAnnotation, BoundingBox, Box, Region
from forte.data.data_pack import DataPack
from forte.common import constants


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
        ip1 = ImagePayload(self.datapack, 0)
        ip1.set_cache(self.line)
        ImageAnnotation(self.datapack)

        self.datapack_1 = DataPack("image_1")
        self.line1 = np.zeros((6, 12))
        self.line1[2, 2] = 1
        self.line1[3, 3] = 1
        self.line1[4, 4] = 1
        ip2 = ImagePayload(self.datapack_1, 0)
        ip2.set_cache(self.line1)
        ImageAnnotation(self.datapack_1)

        self.mark = np.zeros((6, 6))
        self.mark[3, 3:5] = 1
        self.mark[4, 4:5] = 1
        ip2 = ImagePayload(self.datapack_1, 1)
        ip2.set_cache(self.mark)
        ImageAnnotation(self.datapack_1)

        self.datapack_2 = DataPack("image_2")
        self.chunk = np.zeros((12, 12))
        self.chunk[2, :] = 1
        self.chunk[3, :] = 1
        self.chunk[4, :] = 1
        ip3 = ImagePayload(self.datapack_2, 0)
        ip3.set_cache(self.chunk)
        ImageAnnotation(self.datapack_2)

        # Entries for first Data Pack
        self.bb1 = BoundingBox(
            pack=self.datapack_1,
            height=2,
            width=2,
            grid_height=3,
            grid_width=4,
            grid_cell_h_idx=1,
            grid_cell_w_idx=1,
        )
        self.datapack_1.add_entry(self.bb1)

        self.bb2 = BoundingBox(
            pack=self.datapack_1,
            height=3,
            width=4,
            grid_height=5,
            grid_width=5,
            grid_cell_h_idx=3,
            grid_cell_w_idx=3,
        )
        self.datapack_1.add_entry(self.bb2)

        # Entries for another image in the same Data Pack
        self.bb4 = BoundingBox(
            pack=self.datapack_1,
            height=3,
            width=4,
            grid_height=6,
            grid_width=6,
            grid_cell_h_idx=2,
            grid_cell_w_idx=2,
            image_payload_idx=1,
        )
        self.datapack_1.add_entry(self.bb4)

        # Entries for second Data Pack
        self.bb3 = BoundingBox(
            pack=self.datapack_2,
            height=5,
            width=5,
            grid_height=3,
            grid_width=3,
            grid_cell_h_idx=6,
            grid_cell_w_idx=6,
            image_payload_idx=0,
        )
        self.datapack_2.add_entry(self.bb3)

        self.box1 = Box(
            pack=self.datapack_2,
            cy=7,
            cx=7,
            height=4,
            width=2,
            image_payload_idx=0,
        )
        self.datapack_2.add_entry(self.box1)

        self.region1 = Region(pack=self.datapack_2, image_payload_idx=1)
        self.datapack_2.add_entry(self.region1)

    def test_entry_methods(self):
        bb_type = "forte.data.ontology.top.BoundingBox"
        region_type = "forte.data.ontology.top.Region"
        box_type = "forte.data.ontology.top.Box"

        # Analyzing entries in first Data Pack
        bb_list = list(
            self.datapack_1._data_store._DataStore__elements[bb_type]
        )
        bb_entries = list(self.datapack_1._data_store.all_entries(bb_type))

        img1_box_list = self.datapack_1.get_payload_data_at(
            modality=Modality.Image, payload_index=0
        )

        self.assertEqual(bb_list, bb_entries)
        self.assertEqual(
            self.datapack_1._data_store.num_entries(bb_type), len(bb_list)
        )

        self.assertEqual(
            self.datapack.get_single(ImageAnnotation).image_payload_idx, 0
        )

        self.assertEqual(
            len(img1_box_list), 6
        )  # For each bounding box, there is a grid payload created as well

        with self.assertRaises(ProcessExecutionException):
            impossible_box_list = self.datapack_1.get_payload_data_at(
                modality=Modality.Image, payload_index=2
            )

        # Analyzing entries in second Data Pack
        bb_list = list(
            self.datapack_2._data_store._DataStore__elements[bb_type]
        )
        bb_entries = list(self.datapack_2._data_store.all_entries(bb_type))

        box_list = list(
            self.datapack_2._data_store._DataStore__elements[box_type]
        )
        box_entries = list(self.datapack_2._data_store.all_entries(box_type))

        self.assertTrue(
            array_equal(
                self.datapack.get_payload_at(Modality.Image, 0).cache, self.line
            )
        )
        new_pack = DataPack.from_string(self.datapack.to_string())
        self.assertEqual(
            new_pack.audio_annotations, self.datapack.audio_annotations
        )
        region_list = list(
            self.datapack_2._data_store._DataStore__elements[region_type]
        )
        region_entries = list(
            self.datapack_2._data_store.all_entries(region_type)
        )

        # Box and BoundingBox are subclasses of Region
        self.assertEqual(
            len(region_list + box_list + bb_list), len(region_entries)
        )
        self.assertEqual(
            self.datapack_2._data_store.num_entries(region_type),
            len(region_entries),
        )

        # BoundingBox is a subclass of Box
        self.assertEqual(len(box_list + bb_list), len(box_entries))
        self.assertEqual(
            self.datapack_2._data_store.num_entries(box_type), len(box_entries)
        )

    def test_delete_image_annotations(self):

        box_type = "forte.data.ontology.top.BoundingBox"
        box_list = len(
            list(self.datapack_1._data_store._DataStore__elements[box_type])
        )

        self.datapack_1._data_store.delete_entry(self.bb1.tid)
        self.assertEqual(
            self.datapack_1._data_store.num_entries(box_type), box_list - 1
        )

    def test_update_image_annotation(self):
        # Check current value
        self.assertEqual(self.bb1._height, 2)

        # Change a parameter of the entry object
        self.bb1._height = 5

        # Fetch attribute value from data store
        bb1_height = self.datapack_1._data_store.get_attribute(
            self.bb1.tid, "_height"
        )
        # Check new value
        self.assertEqual(bb1_height, 5)

        # Updating Non-Dataclass fields

        # Check current value
        self.assertEqual(self.bb4.image_payload_idx, 1)

        # Change a parameter of the entry object
        self.bb4.image_payload_idx = 2

        # Fetch attribute value from data store
        bb4_payload = self.datapack_1._data_store.get_entry(self.bb4.tid)[0][
            constants.PAYLOAD_INDEX
        ]
        # Check new value
        self.assertEqual(bb4_payload, 2)

    def test_compute_iou(self):
        box1 = self.bb1
        box2 = self.bb2
        box3 = self.bb3
        box4 = self.bb4

        iou1 = box1.compute_iou(box4)
        self.assertEqual(iou1, 0.14285714285714285)

        iou2 = box1.compute_iou(box2)
        self.assertEqual(iou2, 0)

        iou3 = box1.compute_iou(box3)
        self.assertEqual(iou3, 0)

    def test_compute_overlap_from_data_store(self):
        bb1 = self.datapack_1.get_entry(tid=self.bb1.tid)
        bb2 = self.datapack_1.get_entry(tid=self.bb2.tid)

        overlap = bb1.is_overlapped(bb2)
        self.assertTrue(overlap)

    def test_add_image_annotation(self):

        new_box = Box(
            pack=self.datapack_1,
            cy=7,
            cx=7,
            height=4,
            width=2,
            image_payload_idx=0,
        )

        self.assertEqual(
            len(
                self.datapack_1._data_store._DataStore__elements[
                    "forte.data.ontology.top.Box"
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
