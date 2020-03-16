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
Utils for unit tests.
"""

import os
import unittest

__all__ = [
    "performance_test",
    "pretrained_test",
]

from typing import Any, Callable


def define_skip_condition(flag: str, explanation: str):
    return unittest.skipUnless(
        os.environ.get(flag, 0) or os.environ.get('TEST_ALL', 0),
        explanation + f" Set `{flag}=1` or `TEST_ALL=1` to run.")


performance_test = define_skip_condition(
    'TEST_PERFORMANCE', "Test the performance of Forte modules.")

pretrained_test: Callable[[Any], Any] = define_skip_condition(
    'TEST_PRETRAINED', "Test requires loading pre-trained checkpoints.")
