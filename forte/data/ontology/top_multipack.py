from typing import Iterable, Optional, Type, Union, Tuple

from forte.data.ontology.top import Entry, Link, Group


class MultiPackLink(Link):
    """Link type entries, such as "SentencePairLink". Each link has a parent
     node and a child node.
    """
    parent_type: Type[Entry] = Entry  # type: ignore
    child_type: Type[Entry] = Entry  # type: ignore

    def __init__(self,
                 parent: Optional[Entry],
                 child: Optional[Entry],
                 ):
        super().__init__()
        self._parent: Optional[Tuple] = None
        self._child: Optional[Tuple] = None
        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    def set_parent(self, parent: Entry):
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally as a tuple: pack_name and entry.tid
        """
        pack_name = parent.data_pack.meta.name
        assert pack_name is not None, \
            f"The name of the pack {parent.data_pack}"\
            f" including the entry {parent} should not be None"
        if not isinstance(parent, self.parent_type):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.parent_type}, but get {type(parent)}")

        self._parent = (pack_name, parent.tid)

    def set_child(self, child: Entry):
        """
        This will set the `child` of the current instance with given Entry
        The parent is saved internally as a tuple: pack_name and entry.tid
        """
        pack_name = child.data_pack.meta.name
        assert pack_name is not None, \
            f"The name of the pack {child.data_pack}"\
            f" including the entry {child} should not be None"
        if not isinstance(child, self.child_type):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.child_type}, but get {type(child)}")

        self._child = (pack_name, child.tid)

    def get_parent(self):
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link
             from the given DataPack.
        """
        parent_pack_name, parent_tid = self.parent

        if self.data_pack is None:
            raise ValueError(f"Cannot get parent because link is not "
                             f"attached to any data pack.")
        return self.data_pack.packs[parent_pack_name].index.entry_index[
            parent_tid]

    def get_child(self):
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given DataPack
        """
        child_pack_name, child_tid = self.child
        return self.data_pack.packs[child_pack_name].index.entry_index[
            child_tid]


class MultiPackGroup(Group):
    """Group type entries, such as "coreference group". Each group has a set
    of members.
    """

    member_type: Type[Entry] = Entry  # type: ignore

    def add_members(self, members: Union[Iterable[Entry], Entry]):
        """Add group members."""
        if not isinstance(members, Iterable):
            members = {members}

        for member in members:
            if not isinstance(member, self.member_type):
                raise TypeError(
                    f"The members of {type(self)} should be "
                    f"instances of {self.member_type}, but get {type(member)}")
            pack_name = member.data_pack.meta.name
            self._members.add((pack_name, member.tid))

        if (self.data_pack is not None and
                self.data_pack.index.group_index_switch):
            self.data_pack.index.update_group_index([self])

    def get_members(self):
        """
        Get the member entries in the group.

        Returns:
             An set of instances of :class:`Entry` that are the members of the
             group.
        """

        member_entries = set()
        for pack_tid in self.members():
            p, m = pack_tid
            member_entries.add(self.data_pack.packs[p].index.entry_index[m])
        return member_entries
