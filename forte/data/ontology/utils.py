# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for ontology generation.
"""
import os
import sys
from importlib import util as import_util
from pathlib import Path
from pydoc import locate
from typing import Optional, List, Tuple
import json
import jsonschema


def get_user_objects_from_module(module_str: str,
                                 custom_dirs: Optional[str] = None):
    """
    Args:
        module_str: Module in the form of string, package.module.
        custom_dirs: custom directories to search from if `module_str` is not a
          part of imported modules.

    Returns: A list of objects present in the module `module_str`,
             None if module not found

    """
    module = locate(module_str)
    if module is not None and hasattr(module, '__all__'):
        return module.__all__  # type: ignore
    objects: List[str] = []
    if custom_dirs is not None:
        module_file = module_str.replace('.', '/') + '.py'
        for dir_ in custom_dirs:
            filepath = os.path.join(dir_, module_file)
            try:
                spec = import_util.spec_from_file_location(module_str,
                                                           filepath)
                module = import_util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                objects = module.__all__  # type: ignore
            except (FileNotFoundError, AttributeError):
                continue
    return objects


def search_in_dirs(file, dirs_paths):
    """
    Args:
        file: File to be searched for
        dirs_paths: Directory Paths in which the `file` is to be searched

    Returns: Resolved filename if the `file` is found in `dir_paths`,
    else `None`

    """
    abs_file = file
    for _dir in dirs_paths:
        if not os.path.isabs(file):
            abs_file = os.path.join(_dir, file)
        abs_file = str(Path(abs_file).resolve())

        for dir_path in Path(_dir).glob("**/*"):
            resolved_path = str(dir_path.resolve())
            if abs_file == resolved_path:
                return resolved_path
    return None


def get_top_level_dirs(path: Optional[str]):
    """
    Args:
        path: Path for which the directories at depth==1 are to be returned
    Returns:
        Directories at depth==1 for `path`.
    """
    if path is None or not os.path.exists(path):
        return []
    return [item for item in os.listdir(path)
            if os.path.isdir(os.path.join(path, item))]


def split_file_path(path: str):
    """
    Args:
        path: Path to be split

    Returns: list containing path components
    Examples:
        >>> split_file_path('forte/data/ontology/file.py')
        ['forte', 'data', 'ontology', 'file.py']
        >>> split_file_path('/home/file.py')
        ['', 'home', 'file.py']
    """
    path_split = []
    prev_dir, curr_dir = None, (str(Path(path)), '')
    while prev_dir != curr_dir:
        prev_dir = curr_dir
        if curr_dir[-1].strip():
            path_split.append(curr_dir[-1])
        curr_dir = os.path.split(curr_dir[0])
    path_split += [''] if path.startswith('/') else []
    return path_split[::-1]


def validate_json_schema(input_filepath: str, validation_filepath: str):
    """
    Validates the input json schema using validation meta-schema provided in
    `validation_filepath` according to the specification in
    `http://json-schema.org`.
    If the tested json is not valid, a `jsonschema.exceptions.ValidationError`
    is thrown.
    Args:
        input_filepath: Filepath of the json schema to be validated
        validation_filepath: Filepath of the valiodation specification
    """
    with open(validation_filepath, 'r') as validation_json_file:
        validation_schema = json.loads(validation_json_file.read())
    with open(input_filepath, 'r') as input_json_file:
        input_schema = json.loads(input_json_file.read())
    jsonschema.Draft6Validator(validation_schema).validate(input_schema)


def get_python_version() -> Tuple[int, int]:
    """
    :return: Python major and minor version at runtime
    """
    version_info = sys.version_info
    return version_info[0], version_info[1]
