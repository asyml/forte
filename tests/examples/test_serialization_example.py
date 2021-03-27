# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit tests for serialization Example.
"""
import os
import unittest

from examples.serialization import serialize_example


class SerializationExampleTest(unittest.TestCase):
    def test_run_example(self):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../', 'data_samples', 'ontonotes/one_file'))

        serialize_example.main(data_path)

        self.assertTrue(os.path.exists(os.path.join('multi_out', 'multi.idx')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'pack.idx')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'packs')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'multi')))

        self.assertTrue(os.path.exists('pack_out'))

        with open(os.path.join('multi_out', 'multi.idx')) as f:
            for line in f:
                self.assertTrue('/' in line)
                self.assertTrue('\\' not in line)
