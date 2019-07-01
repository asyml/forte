"""
Util functions
"""


def get_full_component_name(o):
    """
    Returns the full module and class name of an object o.
    For example, for our :class: OntonotesReader, returns
    'nlp.pipeline.data.readers.ontonotes_reader.OntonotesReader'.
    """
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__


def get_class_name(o, lower=False):
    """
    Returns the class name of an object o.
    For example, for :class:`OntonotesOntology.Token`, returns
    'Token'.
    """
    if lower:
        return o.__class__.__name__.lower()
    else:
        return o.__class__.__name__


def get_qual_name(o, lower=False):
    """
    Returns the class name of an object o.
    For example, for :class:`OntonotesOntology.Token`, returns
    'OntonotesOntology.Token'.
    """
    if lower:
        return o.__class__.__qualname__.lower()
    else:
        return o.__class__.__qualname__