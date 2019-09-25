from abc import abstractmethod
from typing import TypeVar, Generic
from forte.data.index import BaseIndex

T = TypeVar('T')
IndexType = TypeVar('IndexType', bound=BaseIndex)


class EntryContainer(Generic[T]):
    def __init__(self):
        self.index: IndexType

    @abstractmethod
    def validate(self, item: T) -> bool:
        """
        Validate whether this type can be added.
        Args:
            item:

        Returns:

        """
        raise NotImplementedError

    def get_entry(self, tid: str):
        raise NotImplementedError
