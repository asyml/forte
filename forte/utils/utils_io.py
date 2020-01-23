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
Utility functions related to input/output.
"""
import os

__all__ = [
    "maybe_create_dir",
]


def maybe_create_dir(dirname: str) -> bool:
    r"""Creates directory if it does not exist.

    Args:
        dirname (str): Path to the directory.

    Returns:
        bool: Whether a new directory is created.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        return True
    return False
