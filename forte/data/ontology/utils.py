"""
    Ontology related utils
"""
import os
import inspect
from pydoc import locate

from typing import List, Dict, Union, Type

from forte.data.base_pack import PackType
from forte.data.ontology.base.core import Entry


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


def record_fields(output_info: Dict[Type[Entry], Union[List, Dict]],
                  component_name: str, input_pack: PackType):
    """
    Record the fields and entries that this processor add to packs.

    Args:
        component_name: The component name that do the processing. This will be
        record here to remember which processor (or reader) alter the entries.

        output_info: The output information as specified in a reader of
        process. It contains the information of which types or fields are
        created or altered.

        input_pack: The data pack where the information will be written on.

    Returns:
    """
    for entry_type, info in output_info.items():
        fields: List[str] = []
        if isinstance(info, list):
            fields = info
        elif isinstance(info, dict):
            fields = info["fields"]
            if "component" in info.keys():
                component_name = info["component"]
        input_pack.record_fields(fields, entry_type, component_name)
