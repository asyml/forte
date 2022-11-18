# Copyright 2022 The Forte Authors. All Rights Reserved.
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

import logging
from typing import Any, Dict, Optional, cast
from forte.data.base_pack import PackType
from forte.data.ontology.core import Entry, FList, FDict
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import (
    Annotation,
    Link,
    Group,
    Generics,
    AudioAnnotation,
    ImageAnnotation,
    MultiPackGeneric,
    MultiPackGroup,
    MultiPackLink,
    Payload,
    SinglePackEntries,
    MultiPackEntries,
)
from forte.common import constants
from forte.utils import get_class, get_full_module_name

logger = logging.getLogger(__name__)

__all__ = ["EntryConverter"]


class EntryConverter:
    r"""
    Facilitate the conversion between entry data in list format from
    ``DataStore`` and entry class object.
    """

    def __init__(self) -> None:
        # Mapping from entry's tid to the entry objects for caching
        self._entry_dict: Dict[int, Entry] = {}

    def save_entry_object(
        self,
        entry: Any,
        pack: PackType,
        allow_duplicate: bool = True,
    ):
        # pylint: disable=protected-access
        """
        Save an existing entry object into DataStore.
        """
        # Check if the entry is already stored
        data_store_ref = pack._data_store
        try:
            data_store_ref.get_entry(tid=entry.tid)
            logger.info(
                "The entry with tid=%d is already saved into DataStore",
                entry.tid,
            )
            return
        except KeyError:
            # The entry is not found in DataStore
            pass

        # Create a new registry in DataStore based on entry's type
        if data_store_ref._is_subclass(entry.entry_type(), Annotation):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.BEGIN_ATTR_NAME
                    ),
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.END_ATTR_NAME
                    ),
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), Link):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.PARENT_TYPE_ATTR_NAME
                    ),
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.CHILD_TYPE_ATTR_NAME
                    ),
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), Group):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.MEMBER_TYPE_ATTR_NAME
                    )
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), Generics):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
            )
        elif data_store_ref._is_subclass(entry.entry_type(), AudioAnnotation):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.BEGIN_ATTR_NAME
                    ),
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.END_ATTR_NAME
                    ),
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), ImageAnnotation):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
            )
        elif data_store_ref._is_subclass(entry.entry_type(), Payload):
            entry = cast(Payload, entry)
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
            )
        elif data_store_ref._is_subclass(entry.entry_type(), MultiPackLink):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.PARENT_TYPE_ATTR_NAME
                    ),
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.CHILD_TYPE_ATTR_NAME
                    ),
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), MultiPackGroup):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
                # Once an attribute is accessed from the _cached_attribute_data
                # dict, it must be removed
                attribute_data=[
                    Entry._cached_attribute_data[entry.entry_type()].pop(
                        constants.MEMBER_TYPE_ATTR_NAME
                    )
                ],
            )
        elif data_store_ref._is_subclass(entry.entry_type(), MultiPackGeneric):
            data_store_ref.add_entry_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
            )
        else:
            valid_entries: str = ", ".join(
                map(get_full_module_name, SinglePackEntries + MultiPackEntries)
            )
            raise ValueError(
                f"Invalid entry type {entry.entry_type()}. A valid entry should"
                f" be an instance of {valid_entries}."
            )

        # Store all the dataclass attributes to DataStore
        for attribute, value in Entry._cached_attribute_data[
            entry.entry_type()
        ].items():
            if value is None:
                continue
            if isinstance(value, Entry):
                value = value.tid
            elif isinstance(value, FDict):
                value = {key: val.tid for key, val in value.items()}
            elif isinstance(value, FList):
                value = [val.tid for val in value]
            data_store_ref.set_attribute(
                tid=entry.tid, attr_name=attribute, attr_value=value
            )

        # Empty the cache of the attribute data in Entry
        Entry._cached_attribute_data[entry.entry_type()].clear()
        # Cache the stored entry and its tid
        self._entry_dict[entry.tid] = entry

    def get_entry_object(
        self, tid: int, pack: PackType, type_name: Optional[str] = None
    ) -> EntryType:
        """
        Convert a tid to its corresponding entry object.
        """

        # Check if the tid is cached
        if tid in self._entry_dict:
            return self._entry_dict[tid]  # type: ignore

        data_store_ref = pack._data_store  # pylint: disable=protected-access
        if type_name is None:
            _, type_name = data_store_ref.get_entry(tid=tid)
        entry_class = get_class(type_name)
        entry: Entry
        # pylint: disable=protected-access
        # Here the entry arguments are optional (begin, end, parent, ...) and
        # the value can be arbitrary since they will all be routed to DataStore.
        if data_store_ref._is_annotation(type_name):
            entry = entry_class(pack=pack, begin=0, end=0)
        elif any(
            data_store_ref._is_subclass(type_name, type_class)
            for type_class in SinglePackEntries + MultiPackEntries
        ):
            entry = entry_class(pack=pack)
        else:
            valid_entries: str = ", ".join(
                map(get_full_module_name, SinglePackEntries + MultiPackEntries)
            )
            raise ValueError(
                f"Invalid entry type {type_name}. A valid entry should be an"
                f" instance of {valid_entries}."
            )

        # Remove the new tid and direct the entry object to the correct tid.
        if entry.tid in self._entry_dict:
            self._entry_dict.pop(entry.tid)
        if entry.tid in pack._pending_entries:
            pack._pending_entries.pop(entry.tid)
        data_store_ref.delete_entry(tid=entry.tid)
        entry._tid = tid

        self._entry_dict[tid] = entry
        return entry  # type: ignore
