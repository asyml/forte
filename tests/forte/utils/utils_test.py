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
Unit test for utilities.
"""
import unittest

from ddt import data, ddt

from forte.utils import utils
from forte.utils.utils import DiffAligner


@ddt
class UtilsTest(unittest.TestCase):
    def test_get_full_module_name(self):
        from forte.processors.misc import LowerCaserProcessor

        full_name = utils.get_full_module_name(LowerCaserProcessor)
        self.assertEqual(
            full_name,
            "forte.processors.misc.lowercaser_processor.LowerCaserProcessor",
        )

    def test_get_class_name(self):
        from forte.processors.misc import LowerCaserProcessor

        processor: LowerCaserProcessor = LowerCaserProcessor()
        self.assertEqual(utils.get_class_name(processor), "LowerCaserProcessor")
        self.assertEqual(
            utils.get_class_name(processor, True), "lowercaserprocessor"
        )

    def test_get_class(self):
        cls = utils.get_class(
            "LowerCaserProcessor",
            ["forte.processors.misc.lowercaser_processor"],
        )
        self.assertEqual(cls.__name__, "LowerCaserProcessor")

        with self.assertRaises(ValueError):
            utils.get_class("NonExistProcessor")

        with self.assertRaises(ValueError):
            utils.get_class(
                "NonExistProcessor",
                ["forte.processors.misc.lowercaser_processor"],
            )

    def test_get_qual_name(self):
        from forte.processors.misc import LowerCaserProcessor

        processor: LowerCaserProcessor = LowerCaserProcessor()

        self.assertEqual(utils.get_qual_name(processor), "LowerCaserProcessor")

    def test_create_class_with_kwargs(self):
        p = utils.create_class_with_kwargs(
            class_name="forte.processors.misc.lowercaser_processor"
            ".LowerCaserProcessor",
            class_args={},
        )

        self.assertEqual(
            p.name,
            "forte.processors.misc.lowercaser_processor.LowerCaserProcessor",
        )

    @data(
        (
            "this if  sentences for testing.",
            ["This", "is", "a", "sentence", "4", "testing", "."],
            [(0, 4), (5, 7), None, (9, 17), (19, 22), (23, 30), (30, 31)],
        ),
        ("aa bb   cc", ["aa", "bb", "cc"], [(0, 2), (3, 5), (6, 10)]),
        (
            "aa bb  cc  dd xx yy",
            ["aa", "bb", "cc", "d", "dd", "xx", "yy"],
            [(0, 2), (3, 5), (6, 9), None, (11, 13), (14, 16), (17, 19)],
        ),
    )
    def test_diff_aligner(self, value):
        text, segments, expected_spans = value
        aligner = DiffAligner()
        spans = aligner.align_with_segments(text, segments)
        self.assertEqual(spans, expected_spans)


if __name__ == "__main__":
    unittest.main()
