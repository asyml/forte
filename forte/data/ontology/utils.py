"""
    Ontology related utils
"""
import os
from importlib import util as import_util
from pathlib import Path
from pydoc import locate


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
            resolved_path = str(dir_path.resolve())
            if file == resolved_path:
                return resolved_path
    return None


def get_top_level_dirs(path):
    """
    Args:
        path: Path for which the directories at depth==1 are to be returned
    Returns:
        Directories at depth==1 for `path`.
    """
    if not os.path.exists(path):
        return []
    return [item for item in os.listdir(path)
            if os.path.isdir(os.path.join(path, item))]
