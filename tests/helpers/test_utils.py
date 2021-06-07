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
import filecmp
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


def dir_is_same(dir1, dir2):
    """
        Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.
    Args:
        dir1: First directory path
        dir2: Second directory path

    Returns:
        True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.

    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or \
            len(dirs_cmp.funny_files) > 0:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch) > 0 or len(errors) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not dir_is_same(new_dir1, new_dir2):
            return False
    return True


performance_test = define_skip_condition(
    'TEST_PERFORMANCE', "Test the performance of Forte modules.")

pretrained_test: Callable[[Any], Any] = define_skip_condition(
    'TEST_PRETRAINED', "Test requires loading pre-trained checkpoints.")
