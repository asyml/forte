import copy
import logging
from typing import (Dict, List, Optional, Type, Any)

from forte.data.base_pack import BaseMeta, BasePack
from forte.data import DataPack
from forte.data import Entry, EntryType, Link, Group,\
    MultiPackGroup, MultiPackLink

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackMeta",
    "MultiPack",
    "MultiPackLink",
    "MultiPackGroup",
]


class MultiPackMeta(BaseMeta):
    """
    Meta information of a MultiPack.
    """
    def __init__(self, doc_id: Optional[str] = None):
        super().__init__(doc_id)
        self.span_unit = 'character'


class MultiPack(BasePack):
    """
    A :class:`MultiPack' contains multiple DataPacks and a
    collection of cross-pack entries (annotations, links, and groups)

    Args:
        doc_id (str, optional): A universal id to denote this MultiPack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        super().__init__()
        self.packs: Dict[str, DataPack] = {}
        self.links: List[Link] = []
        self.groups: List[Group] = []
        self.meta: MultiPackMeta = MultiPackMeta(doc_id)

    def update_pack(self, **packs):
        for key, pack in packs.items():
            if key in self.packs:
                raise ValueError(f"{key} is in packs already")
            if not isinstance(key, str):
                raise ValueError(
                    f"key of the pack should be str, but got " f"{type(key)}"
                )
            if not isinstance(pack, DataPack):
                raise ValueError(
                    f"value of the packs should be DataPack, but "
                    f"got {type(pack)}"
                )
            self.packs[key] = pack

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
            name = entry.__class__
            entry.set_tid(str(self.internal_metas[name].id_counter))
            entry.attach(self)
            target.append(entry)

            self.internal_metas[name].id_counter += 1

            # update the data pack index if needed
            self.index.update_basic_index([entry])
            if self.index.link_index_switch and isinstance(entry,
                                                           MultiPackLink):
                self.index.update_link_index([entry])
            if self.index.group_index_switch and isinstance(entry,
                                                            MultiPackGroup):
                self.index.update_group_index([entry])

            return entry
        # logger.debug(f"Annotation already exist {annotation.tid}")
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
        entry.attach(self)
        target.append(entry)
        self.internal_metas[name].id_counter += 1
        return entry

    def view(self):
        return copy.deepcopy(self)

    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        for pack in self.packs.values():
            pack.record_fields(fields, entry_type, component)
