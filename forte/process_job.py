import itertools
from enum import Enum
from typing import Optional

from forte.data.base_pack import PackType

__all__ = [
    "ProcessJobStatus",
    "ProcessJob"
]

ProcessJobStatus = Enum("ProcessJobStatus", "UNPROCESSED QUEUED PROCESSED")


class ProcessJob:
    counter = itertools.count(0)

    def __init__(self, pack: Optional[PackType], is_poison: bool):
        self.__pack: Optional[PackType] = pack
        self.__is_poison: bool = is_poison
        self.__status = ProcessJobStatus.UNPROCESSED
        self.__id = next(ProcessJob.counter)

    def set_status(self, status):
        self.__status = status

    @property
    def id(self):
        return self.__id

    @property
    def pack(self) -> PackType:
        if self.__pack is None:
            raise ValueError("This job do not have a valid pack.")
        return self.__pack

    def alter_pack(self, pack: PackType):
        """
        This class alter the pack in this job. This should only be controlled by
        the system itself. One should not call this function without proper
        understanding.

        Args:
            pack: The pack to be used to replace into this job.

        Returns:

        """
        self.__pack = pack

    @property
    def is_poison(self) -> bool:
        return self.__is_poison

    @property
    def status(self):
        return self.__status
