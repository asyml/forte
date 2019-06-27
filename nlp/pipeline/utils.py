"""
Util functions
"""


def get_full_component_name(o):
    """
    Returns the full module and class name of an object o.
    for example, for our :class: OntonotesReader, returns
    'nlp.pipeline.io.readers.ontonotes_reader.OntonotesReader'
    """
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__


def get_class_name(o, lower=False):
    if lower:
        return o.__class__.__name__.lower()
    else:
        return o.__class__.__name__


def get_qual_name(o, lower=False):
    if lower:
        return o.__class__.__qualname__.lower()
    else:
        return o.__class__.__qualname__