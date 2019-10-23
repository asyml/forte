"""
A utility class containing useful functions for users to interact with the data
packs.
"""
from typing import Type

from forte.data import DataPack
from forte.data.ontology.core import EntryType
from forte.common.exception import EntryNotFoundError


def get_single(pack: DataPack, entry_type: Type[EntryType]) -> EntryType:
    """
    Take a single entry of type :attr:`entry_type` from the provided data pack.
    This is useful when the target entry type normally appears only one time in
    the DataPack. For example, a Document entry.

    Args:
        pack: The provided data pack to take entries from.
        entry_type: The entry type to be retrieved.

    Returns:
        A single data entry.

    """
    for a in pack.get(entry_type):
        return a

    raise EntryNotFoundError(
        f"The entry {entry_type} is not found in the provided data pack.")
