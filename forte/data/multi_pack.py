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

import copy
import logging
from typing import (Dict, List, Set, Union, Iterator, Optional, Type, Any,
                    Tuple)

from sortedcontainers import SortedList

from forte.common import ProcessExecutionException
from forte.data.base_pack import BaseMeta, BasePack
from forte.data.data_pack import DataPack
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import (
    Annotation, MultiPackGroup, MultiPackLink, MultiPackEntries,
    MultiPackGeneric)
from forte.data.span import Span
from forte.data.types import DataRequest

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackMeta",
    "MultiPack",
    "MultiPackLink",
    "MultiPackGroup",
]

MdRequest = Dict[
    Type[Union[MultiPackLink, MultiPackGroup]],
    Union[Dict, List]
]


class MultiPackMeta(BaseMeta):
    r"""Meta information of a MultiPack."""
    pass


# pylint: disable=too-many-public-methods

class MultiPack(BasePack[Entry, MultiPackLink, MultiPackGroup]):
    r"""A :class:`MultiPack' contains multiple DataPacks and a collection of
    cross-pack entries (links, and groups)
    """

    def __init__(self, pack_name: Optional[str] = None):
        super().__init__(pack_name)

        # Store the pack ids of the subpacks. Note that these are UUIDs so
        # they should be globally non-conflicting.
        self._pack_ref: List[int] = []
        # Store the reverse mapping from pack id to the pack index.
        self._inverse_pack_ref: Dict[int, int] = {}

        # Store the pack names.
        self._pack_names: List[str] = []
        # Store the reverse mapping from name to the pack index.
        self._name_index: Dict[str, int] = {}

        # Reference to the real packs.
        self._packs: List[DataPack] = []

        self.links: SortedList[MultiPackLink] = SortedList()
        self.groups: SortedList[MultiPackGroup] = SortedList()
        self.generics: SortedList[MultiPackGeneric] = SortedList()

        # Used to automatically give name to sub packs.
        self.__default_pack_prefix = '_pack'

        self.index: MultiIndex = MultiIndex()

    def __setstate__(self, state):
        r"""In deserialization, we set up the index and the references to the
        data packs inside.
        """
        super().__setstate__(state)

        self.links = SortedList(self.links)
        self.groups = SortedList(self.groups)
        self.generics = SortedList(self.generics)

        self.index = MultiIndex()
        # TODO: index those pointers?
        self.index.update_basic_index(list(self.links))
        self.index.update_basic_index(list(self.groups))
        self.index.update_basic_index(list(self.generics))

        for a in self.links:
            a.set_pack(self)

        for a in self.groups:
            a.set_pack(self)

        for a in self.generics:
            a.set_pack(self)

        # Rebuild the name to index lookup.
        self._name_index = {n: i for (i, n) in enumerate(self._pack_names)}

        # Create the pack list for adding them back.
        self._packs = []

    def __getstate__(self):
        r"""
        Pop some recoverable information in serialization.

        Returns:

        """
        state = super().__getstate__()
        # Do not directly serialize the pack itself.
        state.pop('_packs')

        state['links'] = list(state['links'])
        state['groups'] = list(state['groups'])
        state['generics'] = list(state['generics'])

        return state

    def __iter__(self):
        yield from self.links
        yield from self.groups
        yield from self.generics

    def _init_meta(self, pack_name: Optional[str] = None) -> MultiPackMeta:
        return MultiPackMeta(pack_name)

    def validate(self, entry: EntryType) -> bool:
        return isinstance(entry, MultiPackEntries)

    # TODO: get_subentry maybe useless
    def get_subentry(self, pack_idx: int, entry_id: int):
        return self.get_pack_at(pack_idx).get_entry(entry_id)

    def get_span_text(self, span: Span):
        raise ValueError(
            "MultiPack objects do not contain text, please refer to a "
            "specific data pack to get text.")

    def add_pack(self, ref_name: Optional[str] = None) -> DataPack:
        """
        Create a data pack and add it to this multi pack. If `ref_name` is
        provided, it will be used to index the data pack. Otherwise, a default
        name based on the pack id will be created for this data pack. The
        created data pack will be returned.

        Args:
            ref_name (str): The pack name used to reference this data pack from
              the multi pack.

        Returns: The newly created data pack.

        """
        if ref_name in self._name_index:
            raise ValueError(
                f"The name {ref_name} has already been taken.")
        if ref_name is not None and not isinstance(ref_name, str):
            raise ValueError(
                f"key of the pack should be str, but got "
                f"" f"{type(ref_name)}"
            )

        pack: DataPack = DataPack()
        self.add_pack_(pack, ref_name)
        return pack

    def add_pack_(self, pack: DataPack, ref_name: Optional[str] = None):
        """
        Add a existing data pack to the multi pack.

        Args:
            pack (DataPack): The existing data pack.
            ref_name (str): The name to used in this multi pack.

        Returns:

        """
        if ref_name in self._name_index:
            raise ValueError(
                f"The name {ref_name} has already been taken.")
        if ref_name is not None and not isinstance(ref_name, str):
            raise ValueError(
                f"key of the pack should be str, but got "
                f"" f"{type(ref_name)}"
            )
        if not isinstance(pack, DataPack):
            raise ValueError(
                f"value of the packs should be DataPack, but "
                f"got {type(pack)}"
            )

        pid = pack.meta.pack_id

        if ref_name is None:
            # Create a default name based on the pack id.
            ref_name = f'{self.__default_pack_prefix}_{pid}'

        # Record the pack's global id and names. Also the reverse lookup map.
        self._pack_ref.append(pid)
        self._inverse_pack_ref[pid] = len(self._pack_ref) - 1

        self._pack_names.append(ref_name)
        self._name_index[ref_name] = len(self._pack_ref) - 1

        self._packs.append(pack)

    def get_pack_at(self, index: int) -> DataPack:
        """
        Get data pack at provided index.

        Args:
            index: The index of the pack.

        Returns: The pack at the index.

        """
        # return self._pack_manager.get_from_pool(self._pack_ref[index])
        return self.packs[index]

    def get_pack_index(self, pack_id: int) -> int:
        """
        Get the pack index from the global pack id.

        Args:
            pack_id: The global pack id to find.

        Returns:

        """
        try:
            return self._inverse_pack_ref[pack_id]
        except KeyError as e:
            raise ProcessExecutionException(
                f"Pack {pack_id} is not in this multi-pack.") from e

    def get_pack(self, name: str) -> DataPack:
        """
        Get data pack of name.
        Args:
            name: The name of the pack

        Returns: The pack that has that name.

        """
        return self._packs[self._name_index[name]]

    def pack_ids(self) -> List[int]:
        return self._pack_ref

    @property
    def packs(self) -> List[DataPack]:
        """
        Get the list of Data packs that in the order of added.

        Note that please do not use this

        Returns: List of data packs contained in this multi-pack.

        """
        return self._packs

    @property
    def pack_names(self) -> Set[str]:
        return set(self._pack_names)

    def update_pack(self, named_packs: Dict[str, DataPack]):
        for pack_name, pack in named_packs.items():
            self.add_pack_(pack, pack_name)

    def iter_packs(self) -> Iterator[Tuple[str, DataPack]]:
        for pack_name, pack in zip(self._pack_names, self.packs):
            yield pack_name, pack

    def rename_pack(self, old_name: str, new_name: str):
        r"""Rename the pack to a new name. If the new_name is already taken, a
        ``ValueError`` will be raised. If the old_name is not found, then a
        ``KeyError`` will be raised just as missing value from a dictionary.

        Args:
            old_name: The old name of the pack.
            new_name: The new name to be assigned for the pack.

        Returns:

        """
        if new_name in self._name_index:
            raise ValueError("The new name is already taken.")
        pack_index = self._name_index[old_name]
        self._name_index[new_name] = pack_index
        self._pack_names[pack_index] = new_name

    def iter_groups(self):
        yield from self.groups

    def add_all_remaining_entries(self, component: Optional[str] = None):
        """
        Calling this function will add the entries that are not added to the
        pack manually.

        Args:
            component (str): Overwrite the component record with this.

        Returns:

        """
        super().add_all_remaining_entries(component)
        for pack in self.packs:
            pack.add_all_remaining_entries(component)

    def get_data(self, context_type,
                 request: Optional[DataRequest] = None,
                 skip_k: int = 0) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError(
            "We haven't implemented get data for multi pack data yet.")

    def get_single_pack_data(
            self,
            pack_index: int,
            context_type: Type[Annotation],
            request: Optional[DataRequest] = None,
            skip_k: int = 0
    ) -> Iterator[Dict[str, Any]]:
        r"""Get pack data from one of the packs specified by the name. This is
        equivalent to calling the :meth: `get_data` in :class: `DataPack`.

        Args:
            pack_index (int): The index of a single pack.
            context_type (str): The granularity of the data context, which
                could be any Annotation type.
            request (dict): The entry types and fields required.
                The keys of the dict are the required entry types and the
                value should be either a list of field names or a dict.
                If the value is a dict, accepted items includes "fields",
                "component", and "unit". By setting "component" (a list), users
                can specify the components by which the entries are generated.
                If "component" is not specified, will return entries generated
                by all components. By setting "unit" (a string), users can
                specify a unit by which the annotations are indexed.
                Note that for all annotations, "text" and "span" fields are
                given by default; for all links, "child" and "parent"
                fields are given by default.
            skip_k:Will skip the first k instances and generate
                data from the k + 1 instance.

        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required annotations and context).
        """

        yield from self.get_pack_at(
            pack_index).get_data(context_type, request, skip_k)

    def get_cross_pack_data(
            self,
            request: MdRequest,
    ):
        r"""
        Example:

            .. code-block:: python

                requests = {
                    MultiPackLink:
                        {
                            "component": ["dummy"],
                            "fields": ["speaker"],
                        },
                    base_ontology.Token: ["pos", "sense""],
                    base_ontology.EntityMention: {
                        "unit": "Token",
                    },
                }
                pack.get_cross_pack_data(requests)

        Get data via the links and groups across data packs. The keys could be
        Multipack entries (i.e. MultipackLink and MultipackGroup). The values
        specifies the detailed entry information to be get. The value can be a
        List of field names, then the return result will contains all specified
        fields.

        One can also call this method with more constraints by providing
        a Dict, which can contain the following keys:
          - "fields", this specifies the attribute field names to be obtained
          - "unit", this specifies the unit used to index the annotation
          - "component", this specifies a constraint to take only the entries
          created by the specified component.

        The data request logic is very similar to :meth: ``get_data`` function
        in :class: ``Datapack``, only that this is constraint to the Multipack
        entries.

        Args:
            request: A dict containing the data request. The key is the

        Returns:

        """
        pass

    def __add_entry_with_check(
            self, entry: EntryType, allow_duplicate: bool = True) -> EntryType:
        r"""Internal method to add an :class:`Entry` object to the
        :class:`MultiPack` object.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.
            allow_duplicate (bool): Whether we allow duplicate in the datapack.

        Returns:
            The input entry itself
        """
        if isinstance(entry, MultiPackLink):
            target = self.links
        elif isinstance(entry, MultiPackGroup):
            target = self.groups
        elif isinstance(entry, MultiPackGeneric):
            target = self.generics
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)} for Multipack. A valid "
                f"entry should be an instance of MultiPackLink, MultiPackGroup"
                f", or MultiPackGeneric."
            )

        add_new = allow_duplicate or (entry not in target)

        if add_new:
            target.add(entry)

            # TODO: add the pointers?

            # update the data pack index if needed
            self.index.update_basic_index([entry])
            if self.index.link_index_on and isinstance(entry, MultiPackLink):
                self.index.update_link_index([entry])
            if self.index.group_index_on and isinstance(entry, MultiPackGroup):
                self.index.update_group_index([entry])

            self._pending_entries.pop(entry.tid)
            return entry
        else:
            return target[target.index(entry)]

    def get(self, entry_type: Type[EntryType],  # type: ignore
            components: Optional[Union[str, List[str]]] = None):
        """ Get ``entry_type`` entries from this multi pack.

        Example:

            .. code-block:: python

                for relation in pack.get_entries(
                                    CrossDocEntityRelation,
                                    component=entity_component
                                    ):
                    print(relation.parent)
                    ...

            In the above code snippet, we get entries of type
            ``CrossDocEntityRelation`` within each ``sentence`` which were
            generated by ``entity_component``

        Args:
            entry_type (type): The type of the entries requested.
            components (str or list, optional): The component generating the
                entries requested. If `None`, all valid entries generated by
                any component will be returned.

        Returns:

        """
        # valid type
        valid_id = self.get_ids_by_type(entry_type)
        # valid component
        if components is not None:
            if isinstance(components, str):
                components = [components]
            valid_id &= self.get_ids_by_components(components)

        for entry_id in valid_id:
            yield self.get_entry(entry_id)

    def _add_entry(self, entry: EntryType) -> EntryType:
        r"""Force add an :class:`Entry` object to the :class:`MultiPack` object.

        Allow duplicate entries in a datapack.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.

        Returns:
            The input entry itself
        """
        return self.__add_entry_with_check(entry, True)

    def delete_entry(self, entry: EntryType):
        r"""Delete an :class:`~forte.data.ontology.top.Entry` object from the
         :class:`MultiPack`.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be deleted from the pack.

        """
        if isinstance(entry, MultiPackLink):
            target = self.links
        elif isinstance(entry, MultiPackGroup):
            target = self.groups
        elif isinstance(entry, MultiPackGeneric):
            target = self.generics
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        begin = 0
        for i, e in enumerate(target[begin:]):
            if e.tid == entry.tid:
                target.pop(i + begin)
                break

        # update basic index
        self.index.remove_entry(entry)

        # set other index invalid
        self.index.turn_link_index_switch(on=False)
        self.index.turn_group_index_switch(on=False)

    @classmethod
    def validate_link(cls, entry: EntryType) -> bool:
        return isinstance(entry, MultiPackLink)

    @classmethod
    def validate_group(cls, entry: EntryType) -> bool:
        return isinstance(entry, MultiPackGroup)

    def view(self):
        return copy.deepcopy(self)


class MultiIndex(BaseIndex):
    pass
