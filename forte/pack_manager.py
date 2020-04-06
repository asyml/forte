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
    r""" A global manager that manages global pack information, this manager
    does not assume any pipelines or specific information. This allows some
    information to be passed around the pipeline implicitly.

    At the moment, this pack manager controls the implicit pack IDs (especially
    during serialization and de-serialization). Further, it also controls
    which component is taking control of which data pack.

    """

    default_id_session: int = 0

    # A singleton pattern.
    class __PackManager:
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

            # The token holder is allow to create a pack.
            self.pack_creation_token_holder: Optional[str] = None

            # The pack is obtained by particular component.
            self.obtained_by: Dict[int, str] = {}

            # This creates a re-mapping of some deserialized data packs to
            # their new id.
            self.remap: Dict[int, int] = {}

    __instance: Optional[__PackManager] = None

    def __init__(self):
        if not PackManager.__instance:
            PackManager.__instance = PackManager.__PackManager()

    def get_component(self, pack_id: int) -> Optional[str]:
        return self.instance().obtained_by.get(pack_id, None)

    def obtain_pack(self, pack_id: int, component: str):
        """
        A component call this method to obtain the data pack. Once taken,
        all the modification to the data pack will be recorded by the component.

        Args:
            pack_id:
            component:

        Returns:

        """
        if pack_id not in self.instance().obtained_by:
            self.instance().obtained_by[pack_id] = component
        else:
            raise ProcessFlowException(
                f"Both {component} and {self.instance().obtained_by[pack_id]} "
                f"are trying to obtain the same pack, this is currently "
                f"not allowed.")

    def release_pack(self, pack_id: int):
        """
        A component call this method to release the data pack.

        Args:
            pack_id:

        Returns:

        """
        try:
            self.instance().obtained_by.pop(pack_id)
        except ValueError:
            pass

    def set_remappaed_pack_id(self, pack: ContainerType):
        """
        Give a new id to the pack and remember the remap.

        Args:
            pack:

        Returns:

        """
        # The pack should already have a valid pack id.
        assert get_pack_id(pack) >= 0

        pid = get_pack_id(pack)

        # Record this remapping, and assign a new id to the pack.
        if pid in self.instance().remap:
            raise ProcessFlowException(f"The pack id {pid} "
                                       f"has already been remapped.")

        self.instance().remap[pid] = self.instance().next_id
        pack.meta.pack_id = self.instance().next_id  # type: ignore
        self.instance().next_id += 1

    def get_remapped_id(self, old_id: int) -> int:
        """
        Get the remapped id from the old id.
        Args:
            old_id: The old id.

        Returns: The remapped id.

        """
        return self.instance().remap[old_id]

    def set_pack_id(self, pack: ContainerType):
        """
        Assign the next id to the incoming pack.
        
        Args:
            pack: The pack to assign pack id on.

        Returns:

        """
        # Negative pack id means this is a new pack.
        assert get_pack_id(pack) < 0
        pack.meta.pack_id = self.instance().next_id  # type: ignore
        self.instance().next_id += 1

    def reference_pack(self, pack: ContainerType):
        """
        Add a reference to the data pack or multi pack, so that the pack will
        be kept in the memory. This is similar to counting-based reference
        management, the pack will be released when the count drop to 0.

        As a side effect, if the pack does not have an pid, the system will
        assign an id for it.

        Args:
            pack: The data pack to be register to the system.

        Returns:

        """
        pid: int = get_pack_id(pack)
        # Increment the reference and store the pack itself.
        self.instance().pack_references[pid] += 1
        self.instance().pack_pool[pid] = pack

    def dereference_pack(self, pack: ContainerType):
        """
        This method reduce the count the data pack or multi pack, when the count
        reaches 0, the pack will be released.

        Must remember to de-reference a pack after processing, otherwise
        we will have memory issues.

        Args:
            pack: The pack to de-reference.

        Returns:

        """
        pack_id = get_pack_id(pack)

        if pack_id in self.instance().obtained_by:
            raise ProcessFlowException(
                f"Cannot de-register a pack [{pack_id}] when "
                f"it is still in used by a component "
                f"[{self.instance().obtained_by[pack_id]}]")

        # Reduce the reference count.
        self.instance().pack_references[pack_id] -= 1

        # If the reference count reaches 0, then we can remove the pack from
        # the pool and allow Python to garbage collect it.
        if self.instance().pack_references[pack_id] == 0:
            self.instance().pack_pool.pop(pack_id)

    def get_pack(self, pack_id: int):
        r"""Return the data pack corresponding to the session and id.
        Args:
            pack_id: The pid of this pack.

        Returns:

        """
        return self.instance().pack_pool[pack_id]

    def instance(self) -> __PackManager:
        if self.__instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")

        return self.__instance


def get_pack_id(pack: ContainerType) -> int:
    return pack.meta.pack_id  # type: ignore
