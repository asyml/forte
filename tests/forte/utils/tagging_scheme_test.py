# Copyright 2021 The Forte Authors. All Rights Reserved.
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
Unit test for tagging scheme.
"""
import unittest
from ddt import ddt, data, unpack
from forte.utils.tagging_scheme import bio_merge


@ddt
class TestTaggingScheme(unittest.TestCase):
    def test_bio_merge_success(self):
        tags = ["O", "B", "O", "O", "I", "B", "I", "I", "O", "B", "B", "I"]
        types = [
            "",
            "PER",
            "",
            "",
            "PER",
            "LOC",
            "LOC",
            "LOC",
            "",
            "PER",
            "LOC",
            "LOC",
        ]
        start = [0, 11, 20, 24, 27, 34, 41, 44, 51, 54, 76, 83]
        end = [1, 19, 22, 26, 28, 40, 43, 46, 52, 59, 82, 89]
        indices = list(zip(start, end))

        expected_types, expected_start, expected_end = (
            ["PER", "PER", "LOC", "PER", "LOC"],
            [11, 27, 34, 54, 76],
            [19, 28, 46, 59, 89],
        )
        expected_indices = list(zip(expected_start, expected_end))

        result_types, result_indices = bio_merge(tags, types, indices)

        self.assertEqual(result_types, expected_types)
        self.assertEqual(result_indices, expected_indices)

    @data(["O", "B", "Y"], ["B", "B", "B", "B"])
    def test_invalid_input(self, tags):
        types = ["", "", ""]
        indices = [(0, 1), (2, 3), (4, 5)]

        with self.assertRaises(ValueError):
            bio_merge(tags, types, indices)

    @data(
        (["I", "B", "I", "O", "B", "O"], [(0, 1), (2, 19), (30, 32)]),
        (["O", "I", "B", "I", "O", "I"], [(2, 3), (11, 22), (40, 42)]),
    )
    @unpack
    def test_empty_type(self, tags, expected_indices):
        types = ["", "", "", "", "", ""]
        start = [0, 2, 11, 20, 30, 40]
        end = [1, 3, 19, 22, 32, 42]
        indices = list(zip(start, end))

        expected_types = []
        result_types, result_indices = bio_merge(tags, types, indices)

        self.assertEqual(result_types, expected_types)
        self.assertEqual(result_indices, expected_indices)

    def test_no_leading_B_tag(self):
        tags = ["I", "I", "O", "B", "I", "I"]
        types = ["PER", "LOC", "", "PER", "PER", "LOC"]
        start = [0, 2, 11, 20, 30, 40]
        end = [1, 3, 19, 22, 32, 42]
        indices = list(zip(start, end))

        expected_types = ["PER", "LOC", "PER", "LOC"]
        expected_indices = [(0, 1), (2, 3), (20, 32), (40, 42)]
        result_types, result_indices = bio_merge(tags, types, indices)

        self.assertEqual(result_types, expected_types)
        self.assertEqual(result_indices, expected_indices)
