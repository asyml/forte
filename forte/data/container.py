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
Forte Container module.
"""

# Disable some pylint check for stub and overloads.
# pylint: disable=function-redefined,multiple-statements

from abc import abstractmethod
from typing import Dict, Generic, Set, Tuple, TypeVar, Iterator

__all__ = [
    "EntryContainer",
    "ContainerType",
]

E = TypeVar("E")
L = TypeVar("L")
G = TypeVar("G")


class EntryContainer(Generic[E, L, G]):
    def __init__(self):
        # Record the set of entries created by some components.
        self._creation_records: Dict[str, Set[int]] = {}

        # Record the set of fields modified by this component. The 2-tuple
        # identify the entry field, such as (2, lemma).
        self._field_records: Dict[str, Set[Tuple[int, str]]] = {}

    def __setstate__(self, state):
        r"""In deserialization,
        - The :class:`IdManager` is recreated from the id count.
        """
        self.__dict__.update(state)

        if "creation_records" in self.__dict__:
            self._creation_records = self.__dict__.pop("creation_records")

        if "field_records" in self.__dict__:
            self._field_records = self.__dict__.pop("field_records")

    @abstractmethod
    def on_entry_creation(self, entry: E):
        raise NotImplementedError

    @abstractmethod
    def record_field(self, entry_id: int, field_name: str):
        raise NotImplementedError

    @abstractmethod
    def _validate(self, item: E) -> bool:
        r"""Validate whether this entry type can be added. This method is
        called by the :meth:`~forte.data.ontology.core.Entry.__init__` method
        when an instance of :class:`~forte.data.ontology.core.Entry` is being
        added to the pack.

        Args:
            item: The entry itself.
        """
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, ptr: int) -> E:
        raise NotImplementedError

    def get_span_text(self, begin: int, end: int):
        raise NotImplementedError

    def get_all_creator(self) -> Iterator[str]:
        yield from self._creation_records.keys()


ContainerType = TypeVar("ContainerType", bound=EntryContainer)
