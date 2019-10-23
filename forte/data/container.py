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

    def __init__(self, initial_id_count: int = 0):
        self.__id_counter = initial_id_count

    def get_id(self) -> int:
        i = self.__id_counter
        self.__id_counter += 1
        return i

    def current_id_counter(self) -> int:
        return self.__id_counter


class EntryContainer(Generic[E, L, G]):
    def __init__(self):
        # Record the set of entries created by some components.
        self.creation_records: Dict[str, Set[int]] = {}

        # Record the set of fields modified by this component. The 2-tuple
        # identify the entry field, such as (2, lemma).
        self.field_records: Dict[str, Set[Tuple[int, str]]] = {}

        # The Id manager controls the ID management in this container
        self._id_manager = IdManager()

    def __getstate__(self):
        """
        In serialization:
         - We create a special field for serialization information
         - we don't serialize the IdManager object directly, instead we save
           the max count in the serialization information dict.
        """
        state = self.__dict__.copy()
        state['serialization'] = {}
        # TODO: need test cases here.
        state['serialization']['next_id'] = \
            self._id_manager.current_id_counter()
        state.pop('_id_manager')
        return state

    def __setstate__(self, state):
        """
        In deserialization,
          - The IdManager is recreated from the id count.
        """
        self._id_manager = IdManager(state['serialization']['next_id'])
        self.__dict__.update(state)
        self.__dict__.pop('serialization')

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
