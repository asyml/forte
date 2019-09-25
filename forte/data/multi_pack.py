import copy
import logging
from typing import (Dict, List, Union, Iterator, Optional, Type, Any, Tuple)

from forte.data.base_pack import BaseMeta, BasePack
from forte.data.data_pack import DataPack, DataRequest
from forte.data.ontology import Entry, Annotation, MultiPackGroup, MultiPackLink
from forte.common.types import EntryType

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
    """
    Meta information of a MultiPack.
    """

    def __init__(self):
        super().__init__()


class MultiPack(BasePack):
    """
    A :class:`MultiPack' contains multiple DataPacks and a
    collection of cross-pack entries (annotations, links, and groups)
    """

    def __init__(self):
        super().__init__()
        self._packs: List[DataPack] = []
        self._pack_names: List[str] = []
        self.__name_index = {}

        self.links: List[MultiPackLink] = []
        self.groups: List[MultiPackGroup] = []
        self.meta: MultiPackMeta = MultiPackMeta()

        self.__default_pack_prefix = '_pack'

    def add_pack(self, pack: DataPack, pack_name: Optional[str] = None):
        if pack_name in self.__name_index:
            raise ValueError(
                f"The name {pack_name} has already been taken.")
        if pack_name is not None and not isinstance(pack_name, str):
            raise ValueError(
                f"key of the pack should be str, but got "
                f"" f"{type(pack_name)}"
            )
        if not isinstance(pack, DataPack):
            raise ValueError(
                f"value of the packs should be DataPack, but "
                f"got {type(pack)}"
            )

        self._packs.append(pack)
        pid = len(self._pack_names) - 1

        if pack_name is None:
            pack_name = f'{self.__default_pack_prefix}_{pid}'

        self._pack_names.append(pack_name)
        self.__name_index[pack_name] = pid

    @property
    def packs(self) -> List[DataPack]:
        return self._packs

    def update_pack(self, named_packs: Dict[str, DataPack]):
        for pack_name, pack in named_packs.items():
            self.add_pack(pack, pack_name)

    def iter_packs(self) -> Iterator[Tuple[str, DataPack]]:
        for pack_name, pack in zip(self._pack_names, self._packs):
            yield pack_name, pack

    def rename_pack(self, old_name: str, new_name: str):
        """
        Rename the pack to a new name. If the new_name is already taken, a
        ValueError will be raised. If the old_name is not found, then a KeyError
        will be raised just as missing value from a dictionary.

        Args:
            old_name: The old name of the pack.
            new_name: The new name to be assigned for the pack.

        Returns:

        """
        if new_name in self.__name_index:
            raise ValueError("The new name is already taken.")
        pack_index = self.__name_index[old_name]
        self.__name_index[new_name] = pack_index
        self._pack_names[pack_index] = new_name

    def get_pack(self, name: str):
        return self._packs[self.__name_index[name]]

    def get_single_pack_data(
            self,
            pack_index: int,
            context_type: Type[Annotation],
            request: Optional[DataRequest] = None,
            skip_k: int = 0
    ) -> Iterator[Dict[str, Any]]:
        """
        Get pack data from one of the packs specified by the name. This is
        equivalent to calling the :meth: `get_data` in :class: `DataPack`.

        Args:
            pack_index (str): The name to identify the single pack.
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

        yield from self._packs[
            pack_index].get_data(context_type, request, skip_k)

    def get_cross_pack_data(
            self,
            request: MdRequest,
    ):
        """
        Example:

            .. code-block:: python

                requests = {
                    MultiPackLink:
                        {
                            "component": ["dummy"],
                            "fields": ["speaker"],
                        },
                    base_ontology.Token: ["pos_tag", "sense""],
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

    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`Entry` object to the :class:`Multipack` object.
        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries to be
        same if their :meth:`eq` have the same return value, and users could
        override :meth:`eq` in their custom entry classes.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        if isinstance(entry, MultiPackLink):
            target: List[Any] = self.links
        elif isinstance(entry, MultiPackGroup):
            target = self.groups
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        if entry not in target:
            # add the entry to the target entry list
            entry_cls = entry.__class__
            entry.set_tid(str(self.internal_metas[entry_cls].id_counter))
            target.append(entry)

            self.internal_metas[entry_cls].id_counter += 1

            # update the data pack index if needed
            self.index.update_basic_index([entry])
            if self.index.link_index_switch and isinstance(
                    entry, MultiPackLink):
                self.index.update_link_index([entry])
            if self.index.group_index_switch and isinstance(
                    entry, MultiPackGroup):
                self.index.update_group_index([entry])

            return entry
        return target[target.index(entry)]

    def add_entry(self, entry: EntryType) -> EntryType:
        """
        Force add an :class:`Entry` object to the :class:`MultiPack` object.
        Allow duplicate entries in a datapack.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.

        Returns:
            The input entry itself
        """
        if isinstance(entry, MultiPackLink):
            target: List[Any] = self.links
        elif isinstance(entry, MultiPackGroup):
            target = self.groups
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        # add the entry to the target entry list
        name = entry.__class__
        entry.set_tid(str(self.internal_metas[name].id_counter))
        target.append(entry)
        self.internal_metas[name].id_counter += 1
        return entry

    def view(self):
        return copy.deepcopy(self)

    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        for pack in self._packs:
            pack.record_fields(fields, entry_type, component)
