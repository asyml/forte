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
Unit tests for Resources.
"""
import unittest
import tempfile
import shutil

from forte.common.resources import Resources


class DummyObject:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if self.a == other.a and self.b == other.b:
            return True
        return False


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            '1': 'one',
            'dummy': DummyObject(1, 2)
        }
        self.resources = Resources(**self.kwargs)
        self.output_dir = tempfile.mkdtemp()

    def test_save_with_keys(self):
        keys = list(self.kwargs.keys())
        self.resources.save(keys=keys, output_dir=self.output_dir)

        new_resources = Resources()
        new_resources.load(keys=keys, path=self.output_dir)

        self.assertEqual(new_resources._resources, self.resources._resources)

    def test_save_without_keys(self):
        self.resources.save(output_dir=self.output_dir)

        new_resources = Resources()
        keys = list(self.kwargs.keys())
        new_resources.load(keys=keys, path=self.output_dir)

        self.assertEqual(new_resources._resources, self.resources._resources)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)


if __name__ == '__main__':
    unittest.main()
