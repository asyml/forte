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
Forte Span module.
"""

from functools import total_ordering

__all__ = [
    "Span",
]


@total_ordering
class Span:
    r"""A class recording the span of annotations. :class:`Span` objects can
    be totally ordered according to their :attr:`begin` as the first sort key
    and :attr:`end` as the second sort key.

    Args:
        begin (int): The offset of the first character in the span.
        end (int): The offset of the last character in the span + 1. So the
            span is a left-closed and right-open interval ``[begin, end)``.
    """

    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end

    def __repr__(self):
        return f'({self.begin}, {self.end})'

    def __lt__(self, other):
        if self.begin == other.begin:
            return self.end < other.end
        return self.begin < other.begin

    def __eq__(self, other):
        return (self.begin, self.end) == (other.begin, other.end)
