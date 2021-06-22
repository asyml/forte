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
from ddt import ddt, data
from forte.utils.tagging_scheme import bio_merge

@ddt
class TestTaggingScheme(unittest.TestCase):
    def test_bio_merge_success(self):
        tags = ['O', 'B', 'O', 'O', 'I', 'B', 'I', 'I', 'O', 'B', 'B', 'I']
        types = ['', 'PER', '', '', 'PER', 'LOC', 'LOC', 'LOC', '', 'PER',
                 'LOC', 'LOC']
        start = [0, 11, 20, 24, 27, 34, 41, 44, 51, 54, 76, 83]
        end = [1, 19, 22, 26, 28, 40, 43, 46, 52, 59, 82, 89]

        expected_type, expected_start, expected_end = \
            ['PER', 'PER', 'LOC', 'PER', 'LOC'], \
            [11, 27, 34, 54, 76], \
            [19, 28, 46, 59, 89]

        result_type, result_start, result_end = bio_merge(tags, types, start, end)

        self.assertEqual(result_type, expected_type)
        self.assertEqual(result_start, expected_start)
        self.assertEqual(result_end, expected_end)

    @data(['O', 'B', 'Y'], ['B', 'B', 'B', 'B'])
    def test_invalid_input(self, tags):
        types = [None, None, None]

        with self.assertRaises(ValueError):
            bio_merge(tags, types)
