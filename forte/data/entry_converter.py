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
from typing import Dict
from forte.common.constants import TID_INDEX
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
    Grids,
    MultiPackGeneric,
    MultiPackGroup,
    MultiPackLink,
)
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
        self, entry: Entry, pack: PackType, allow_duplicate: bool = True
    ):
        """
        Save an existing entry object into DataStore.
        """
        # Check if the entry is already stored
        data_store_ref = pack._data_store  # pylint: disable=protected-access
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
        if isinstance(entry, Annotation):
            data_store_ref.add_annotation_raw(
                type_name=entry.entry_type(),
                begin=entry.begin,
                end=entry.end,
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
            )
        elif isinstance(entry, Link):
            data_store_ref.add_link_raw(
                type_name=entry.entry_type(),
                parent_tid=entry.parent,
                child_tid=entry.child,
                tid=entry.tid,
            )
        elif isinstance(entry, Group):
            data_store_ref.add_group_raw(
                type_name=entry.entry_type(),
                member_type=get_full_module_name(entry.MemberType),
                tid=entry.tid,
            )
        elif isinstance(entry, Generics):
            data_store_ref.add_generics_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
            )
        elif isinstance(entry, AudioAnnotation):
            data_store_ref.add_audio_annotation_raw(
                type_name=entry.entry_type(),
                begin=entry.begin,
                end=entry.end,
                tid=entry.tid,
                allow_duplicate=allow_duplicate,
            )
        elif isinstance(entry, ImageAnnotation):
            data_store_ref.add_image_annotation_raw(
                type_name=entry.entry_type(),
                image_payload_idx=entry.image_payload_idx,
                tid=entry.tid,
            )
        elif isinstance(entry, Grids):
            data_store_ref.add_grid_raw(
                type_name=entry.entry_type(),
                image_payload_idx=entry.image_payload_idx,
                tid=entry.tid,
            )
        elif isinstance(entry, MultiPackLink):
            data_store_ref.add_multipack_link_raw(
                type_name=entry.entry_type(),
                parent_pack_id=entry.parent[0],
                parent_tid=entry.parent[1],
                child_pack_id=entry.child[0],
                child_tid=entry.child[1],
                tid=entry.tid,
            )
        elif isinstance(entry, MultiPackGroup):
            data_store_ref.add_multipack_group_raw(
                type_name=entry.entry_type(),
                member_type=get_full_module_name(entry.MemberType),
                tid=entry.tid,
            )
        elif isinstance(entry, MultiPackGeneric):
            data_store_ref.add_multipack_generic_raw(
                type_name=entry.entry_type(),
                tid=entry.tid,
            )
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, Group, Generics "
                "or AudioAnnotation."
            )

        # Store all the dataclass attributes to DataStore
        for attribute in entry.__dataclass_fields__:
            value = getattr(entry, attribute, None)
            if not value:
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

        # Cache the stored entry and its tid
        self._entry_dict[entry.tid] = entry

    def get_entry_object(self, tid: int, pack: PackType) -> EntryType:
        """
        Convert a tid to its corresponding entry object.
        """

        # Check if the tid is cached
        if tid in self._entry_dict:
            return self._entry_dict[tid]  # type: ignore

        data_store_ref = pack._data_store  # pylint: disable=protected-access
        entry_data, entry_type = data_store_ref.get_entry(tid=tid)
        entry_class = get_class(entry_type)
        entry: Entry
        # Here the entry arguments are optional (begin, end, parent, ...) and
        # the value can be arbitrary since they will all be routed to DataStore.
        if issubclass(entry_class, (Annotation, AudioAnnotation)):
            entry = entry_class(pack=pack, begin=0, end=0)
        elif issubclass(
            entry_class,
            (
                Link,
                Group,
                Generics,
                MultiPackGeneric,
                MultiPackGroup,
                MultiPackLink,
            ),
        ):
            entry = entry_class(pack=pack)
        else:
            raise ValueError(
                f"Invalid entry type {type(entry_class)}. A valid entry "
                f"should be an instance of Annotation, Link, Group, Generics "
                "or AudioAnnotation."
            )

        # TODO: Remove the new tid and direct the entry object to the correct
        # tid. The implementation here is a little bit hacky. Will need a stable
        # solution in future.
        # pylint: disable=protected-access
        if entry.tid in self._entry_dict:
            self._entry_dict.pop(entry.tid)
        if entry.tid in pack._pending_entries:
            pack._pending_entries.pop(entry.tid)
        data_store_ref.delete_entry(tid=entry.tid)
        entry._tid = entry_data[TID_INDEX]

        self._entry_dict[tid] = entry
        return entry  # type: ignore
