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
import threading
from typing import Dict, Optional, Tuple

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

            # Here the mapping is designed to be a two-int tuple look up key to
            # the data pack.
            # The first is a session ID that is used to differentiate
            # the ID count space caused by different readers. The second is the
            # counter in that count space.

            self.pack_pool: Dict[Tuple[int, int], ContainerType] = {}

            # This dictionary contains the flattened id map. We can find the
            # actual global index of from a packs 2-int tuple look up key.
            self.global_id_map: Dict[Tuple[int, int], int] = {}

            self.global_next_id: int = 0
            self.default_next_id: int = 0
            self.pack_id_session: int = PackManager.default_id_session

            # The pack is obtained by particular component.
            self.locked_pack: Dict[Tuple[int, int], str] = {}

    instance: Optional[__PackManager] = None

    def __init__(self):
        if not PackManager.instance:
            PackManager.instance = PackManager.__PackManager()
        self.__lock = threading.Lock()

    def get_global_id(self, session_id: int, pack_id: int) -> int:
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")
        return self.instance.global_id_map[(session_id, pack_id)]

    def get_component(self, session_id: int, pack_id: int) -> str:
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")

        if (session_id, pack_id) in self.instance.locked_pack:
            return self.instance.locked_pack[(session_id, pack_id)]
        else:
            raise ProcessFlowException(
                f"The pack indexed by [{session_id}, {pack_id}] "
                f"has not obtained by any component.")

    def lock_pack(self, session_id: int, pack_id: int, component: str):
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")

        with self.__lock:
            self.instance.locked_pack[(session_id, pack_id)] = component

    def release_pack(self, session_id: int, pack_id: int):
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")
        with self.__lock:
            try:
                self.instance.locked_pack.pop((session_id, pack_id))
            except ValueError:
                pass

    def get_new_session(self) -> int:
        """Call this when the Pack ID may collide with the existing pack
        ids, this will open up a new session, a separate space for the IDs.
        The major use case is when de-serializing multiple independently
        serialized packs. Each deserializer may need a new session here.

        Returns: The session Id to be used.
        """
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")
        with self.__lock:
            self.instance.pack_id_session += 1
            return self.instance.pack_id_session

    def deregister_pack(self, session_id: int, pack_id: int):
        """
        Must remember to de-register a pack after processing, otherwise
        we will have memory issues.
        Args:
            session_id:
            pack_id:

        Returns:

        """
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")
        if (session_id, pack_id) in self.instance.locked_pack:
            raise ProcessFlowException(
                f"Cannot de-register a pack [{session_id},{pack_id}] when "
                f"it is still in used by a component "
                f"[{self.instance.locked_pack[(session_id, pack_id)]}]")
        self.instance.pack_pool.pop((session_id, pack_id))

    def register_pack(self, pack: ContainerType):
        """
        Register a data pack, using the default session counter.

        Args:
            pack: the pack itself.

        Returns: The pid of the added pack.

        """
        return self.register_pack_with_session(
            PackManager.default_id_session, pack)

    def register_pack_with_session(
            self, id_session: int, pack: ContainerType):
        """
        Add a data pack or multi pack to the pool. If this is called via the
        new instance creation, then it won't have a id, the system will assign
        an id for it.

        Args:
            id_session: The ID session indicates id space of this pack.
            pack: The pack itself.

        Returns:

        """
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")

        with self.__lock:
            meta = pack.meta  # type: ignore

            meta.serial_session = id_session

            s_pid: Tuple[int, int]
            if meta.pack_id < 0:
                # This is a new pack, new pid will be assigned.
                self.instance.default_next_id += 1
                s_pid = (id_session, self.instance.default_next_id)
                meta.pack_id = self.instance.default_next_id
            else:
                s_pid = (id_session, meta.pack_id)
                if s_pid in self.instance.pack_pool:
                    return

            self.instance.pack_pool[s_pid] = pack
            self.instance.global_next_id += 1
            self.instance.global_id_map[s_pid] = self.instance.global_next_id

    def get_pack(self, id_session: int, pack_id: int):
        r"""Return the data pack corresponding to the session and id.
        Args:
            id_session: The id session that PIDs are assigned
            pack_id: The pid of this pack.

        Returns:

        """
        if self.instance is None:
            raise ProcessFlowException("The pack manager is not initialized.")

        return self.instance.pack_pool[(id_session, pack_id)]
