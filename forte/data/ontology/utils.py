"""
    Ontology related utils
"""
import os
import sys
from importlib import util as import_util
from pathlib import Path
from pydoc import locate

from typing import List, Dict, Union, Type

from forte.data.base_pack import PackType
from forte.data.ontology.base.core import Entry


def get_user_objects_from_module(module_str, custom_dirs=None):
    """
    Args:
        module_str: Module in the form of string, package.module.
        custom_dirs: custom directories to search from if `module_str` not a
        part of imported modules. 
        
    Returns: A list of objects present in the module `module_str`,
             None is module not found

    """
    module = locate(module_str)
    if module is None:
        if custom_dirs is not None:
            module_file = module_str.replace('.', '/') + '.py'
            for dir_ in custom_dirs:
                filepath = os.path.join(dir_, module_file)
                try:
                    spec = import_util.spec_from_file_location(module_str,
                                                               filepath)
                    module = import_util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                return module.__all__
        return []
    return module.__all__


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


def search_in_dirs(file, dirs_paths):
    """
    Args:
        file: File to be searched for
        dirs_paths: Directory Paths in which the `file` is to be searched

    Returns: Resolved filename if the `file` is found in `dir_paths`,
    else `None`

    """
    for _dir in dirs_paths:
        if not os.path.isabs(file):
            file = os.path.join(_dir, file)
        file = str(Path(file).resolve())
        for dir_path in Path(_dir).glob("**/*"):
            if file == str(dir_path.resolve()):
                return file
    return None
