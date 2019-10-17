from abc import abstractmethod
from typing import TypeVar, Generic, Dict, Set, Tuple

from forte.data.base import Span
from forte.process_manager import ProcessManager

E = TypeVar('E')
L = TypeVar('L')
G = TypeVar('G')

process_manager = ProcessManager()


class IdManager:
    """
    Control the ids assigned to each entry.
    """

    def __init__(self):
        self.__id_counter = 0

    def get_id(self):
        i = self.__id_counter
        self.__id_counter += 1
        return i


class EntryContainer(Generic[E, L, G]):
    def __init__(self):
        # Record the set of entries created by some components.
        self.creation_records: Dict[str, Set[int]] = {}

        # Record the set of fields modified by this component. The 2-tuple
        # identify the entry field, such as (2, lemma).
        self.field_records: Dict[str, Set[Tuple[int, str]]] = {}

        # The Id manager controls the ID management in this container
        self._id_manager = IdManager()

    def add_entry_creation_record(self, entry_id: int):
        c = process_manager.component

        try:
            self.creation_records[c].add(entry_id)
        except KeyError:
            self.creation_records[c] = {entry_id}

    def add_field_record(self, entry_id: int, field_name: str):
        c = process_manager.component
        try:
            self.field_records[c].add((entry_id, field_name))
        except KeyError:
            self.field_records[c] = {(entry_id, field_name)}

    @abstractmethod
    def validate(self, item: E) -> bool:
        """
        Validate whether this entry type can be added. This method is called by
        the entries at the init stage.

        Args:
            item: The entry itself.
        Returns:
        """
        raise NotImplementedError

    def get_entry(self, tid: int):
        raise NotImplementedError

    def get_span_text(self, span: Span):
        raise NotImplementedError

    def get_next_id(self):
        return self._id_manager.get_id()


ContainerType = TypeVar("ContainerType", bound=EntryContainer)
