"""
    Ontology related utils
"""
import os
import inspect
from pydoc import locate


def get_classes_from_module(module_str):
    """
    Args:
        module_str: Module in the form of string, package.module.

    Returns: A list of classes present in the module `module_str`

    """
    module = locate(module_str)
    classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj):
            classes.append(f"{obj.__name__}")
    return classes


def get_classes_from_folder(directory):
    """
    Args:
        directory: Directory path

    Returns: A list of classes present in the modules contained in the directory

    """
    import glob
    modules = glob.glob(os.path.join(directory, "*.py"))
    classes = [get_classes_from_module(module) for module in modules]
    return classes
