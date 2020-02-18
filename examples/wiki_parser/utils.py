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
Utility functions.
"""
from typing import Type

from forte.data import DataPack
from forte.data.ontology.core import EntryType
from forte.common.exception import EntryNotFoundError

__all__ = [
    "get_single",
]


def get_single(pack: DataPack, entry_type: Type[EntryType]) -> EntryType:
    r"""Take a single entry of type :attr:`entry_type` from the provided data
    pack. This is useful when the target entry type normally appears only one
    time in the :class:`DataPack` for e.g., a Document entry.

    Args:
        pack: The provided data pack to take entries from.
        entry_type: The entry type to be retrieved.

    Returns:
        A single data entry.
    """
    for a in pack.get(entry_type):
        return a

    raise EntryNotFoundError(
        f"The entry {entry_type} is not found in the provided data pack.")
