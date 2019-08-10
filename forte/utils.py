"""
Util functions
"""
from pydoc import locate

__all__ = [
    "get_full_module_name",
    "get_class_name",
    "get_class",
    "get_qual_name",
]


def get_full_module_name(o, lower=False):
    """
    Returns the full module and class name of an object o.
    For example, for our :class: OntonotesReader, returns
    'nlp.forte.data.readers.ontonotes_reader.OntonotesReader'.
    """
    if not isinstance(o, type):
        o = o.__class__
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__
    name = module + '.' + o.__name__
    if lower:
        return name.lower()
    else:
        return name


def get_class_name(o, lower=False):
    """
    Returns the class name of an object o.
    For example, for :class:`OntonotesOntology.Token`, returns
    'Token'.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__name__.lower()
    else:
        return o.__name__


def get_class(class_name, module_paths=None):
    """Returns the class based on class name. (brought from Texar)

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_


def get_qual_name(o, lower=False):
    """
    Returns the class name of an object o.
    For example, for :class:`OntonotesOntology.Token`, returns
    'OntonotesOntology.Token'.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__qualname__.lower()
    else:
        return o.__qualname__
