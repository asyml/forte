from abc import abstractmethod
from typing import TypeVar, Generic
from forte.data.base import Span

E = TypeVar('E')
L = TypeVar('L')
G = TypeVar('G')


class EntryContainer(Generic[E, L, G]):
    def __init__(self):
        pass

    @abstractmethod
    def validate(self, item: E) -> bool:
        """
        Validate whether this type can be added.
        Args:
            item:

        Returns:

        """
        raise NotImplementedError

    def get_entry(self, tid: str):
        raise NotImplementedError

    def get_span_text(self, span: Span):
        raise NotImplementedError
