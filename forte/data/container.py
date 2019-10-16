from abc import abstractmethod
from typing import TypeVar, Generic, Dict, Set, Tuple

from forte.data.base import Span

E = TypeVar('E')
L = TypeVar('L')
G = TypeVar('G')


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
        # This is used internally when a processor takes the ownership of this
        # DataPack.
        self._owner_component: str = '__default__'
        self.component_records: Dict[
            str,  # The component name.
            Set[int],  # The set of entries created by this component.
            Set[  # The set of fields modified by this component.
                Tuple[int, str]  # The 2-tuple identify the entry field.
            ]
        ]
        self._id_manager = IdManager()

    def enter_processing(self, component_name: str):
        self._owner_component = component_name

    def current_component(self):
        return self._owner_component

    def exit_processing(self):
        self._owner_component = '__default__'

    def add_entry_creation_record(self, component_name: str, entry_id: int):
        self.internal_metas['']

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
