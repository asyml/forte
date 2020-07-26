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
from collections import Counter
from typing import Dict, Optional

from forte.common.exception import ProcessFlowException
from forte.data.container import ContainerType


class PackManager:
    r""" A manager that manages global pack information.

    This pack manager controls the implicit pack IDs (especially
    during serialization and de-serialization). Further, it also controls
    which component is taking control of which data pack.

    """

    def __init__(self):
        # In this attribute we store a mapping from a unique identifier
        # to the data packs. However, importing DataPack or even PackType
        # will create cyclic imports, so we only import ContainerType.
        #
        # This pool is used to hold the packs that need to be used. In
        # a single pack case, the pack can be removed once used. In a multi
        # pack case, this can hold the packs until the life cycle of the
        # multi pack. Note that if the pack pool is not release, there may
        # be memory leakage.
        self.pack_references: Counter = Counter()
        self.pack_pool: Dict[int, ContainerType] = {}

        # A global ID counter.
        self.next_id: int = 0

        # This is the initial reader component.
        self.initial_reader: Optional[str] = None

        # This creates a re-mapping of some deserialized data packs to
        # their new id.
        self.remap: Dict[int, int] = {}

    def reset(self):
        self.__init__()

    def reset_remap(self):
        """
        A reader may call this to clear the remap dictionary.

        Returns:

        """
        self.remap.clear()

    def set_remapped_pack_id(self, pack: ContainerType):
        """
        Give a new id to the pack and remember the remap.

        Args:
            pack:

        Returns:

        """
        # The pack should already have a valid pack id.
        assert get_pack_id(pack) >= 0

        pid = get_pack_id(pack)

        # Assign a new id to the pack.
        if pid not in self.remap:
            self.remap[pid] = self.next_id

        pack.meta.pack_id = self.remap[pid]  # type: ignore
        self.next_id += 1

    def get_remapped_id(self, old_id: int) -> int:
        """
        Get the remapped id from the old id.

        Args:
            old_id: The old id.

        Returns: The remapped id. -1 if not found.

        """
        return self.remap.get(old_id, -1)

    def set_pack_id(self, pack: ContainerType):
        """
        Assign the next id to the incoming pack.

        Args:
            pack: The pack to assign pack id on.

        Returns:

        """
        # Negative pack id means this is a new pack.
        assert get_pack_id(pack) < 0
        pack.meta.pack_id = self.next_id  # type: ignore
        self.next_id += 1

    def reference_pack(self, pack: ContainerType):
        """
        Add a reference to the data pack or multi pack, so that the pack will
        be kept in the memory. This is similar to counting-based reference
        management, the pack will be released when the count drop to 0.

        As a side effect, if the pack does not have an id, the system will
        assign an id for it.

        Args:
            pack: The data pack to be register to the system.

        Returns:

        """
        pid: int = get_pack_id(pack)
        # Increment the reference and store the pack itself.
        self.pack_references[pid] += 1
        self.pack_pool[pid] = pack

    def dereference_pack(self, pack_id: int):
        """
        This method reduce the count the data pack or multi pack, when the count
        reaches 0, the pack will be released.

        Must remember to de-reference a pack after processing, otherwise
        we will have memory issues.

        Args:
            pack_id: The pack id that points to the pack to be de-referenced.

        Returns:

        """
        if pack_id not in self.pack_references:
            # This can happen when the instance is reset by the pipeline.
            return

        if self.pack_references[pack_id] < 0:
            # I am not sure if there are cases that can deduct the reference
            # count too much, but we'd put a check here just in case.
            raise ProcessFlowException(
                f"Pack reference count for pack [{pack_id}] is only "
                f"{self.pack_references[pack_id]},"
                f" which is invalid.")

        # Reduce the reference count.
        self.pack_references[pack_id] -= 1

        # If the reference count reaches 0, then we can remove the pack from
        # the pool and allow Python to garbage collect it.
        if self.pack_references[pack_id] == 0:
            self.pack_pool.pop(pack_id)

    def get_from_pool(self, pack_id: int) -> ContainerType:
        r"""Return the data pack corresponding to the id.
        Args:
            pack_id: The pid of this pack.

        Returns: The pack indexed by this pack id.

        """
        return self.pack_pool[pack_id]


def get_pack_id(pack: ContainerType) -> int:
    return pack.meta.pack_id  # type: ignore
