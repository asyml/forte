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
Unit test for Span.
"""

import unittest

from forte.data.span import Span


class SpanTest(unittest.TestCase):
    def test_span(self):
        span1 = Span(1, 2)
        span2 = Span(1, 2)
        self.assertEqual(span1, span2)

        span1 = Span(1, 2)
        span2 = Span(1, 3)
        self.assertLess(span1, span2)

        span1 = Span(1, 2)
        span2 = Span(2, 3)
        self.assertLess(span1, span2)


if __name__ == "__main__":
    unittest.main()
