class PackIndexError(Exception):
    """
    Raise this error when there is a problem accessing the indexes pack data.
    """
    pass


class IncompleteEntryError(Exception):
    """
    Raise this error when the entry is not complete
    """
    pass
