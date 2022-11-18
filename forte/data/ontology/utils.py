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
import re
import json
import fnmatch

from pathlib import Path
from pydoc import locate
from importlib import util as import_util
from typing import Optional, List, Tuple

from distutils.file_util import copy_file
from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
from distutils import log

import jsonschema

AUTO_GEN_SIGNATURE = "***automatically_generated***"


def copytree(
    src,
    dst,
    ignore_pattern_if_file_exists="*",
    preserve_mode=True,
    preserve_times=False,
    preserve_symlinks=False,
    update=False,
    verbose=1,
    dry_run=0,
):
    """
    A slightly modified version of `distutils.dir_util.copytree`, with the
    added parameter `ignore_pattern_if_file_exists` that ignores files if it
    matches the given pattern and already exists in the destination directory.
    By default, ignores all existing destination files.
    """

    if not dry_run and not os.path.isdir(src):
        raise DistutilsFileError("cannot copy tree '%s': not a directory" % src)
    try:
        names = os.listdir(src)
    except OSError as e:
        if dry_run:
            names = []
        else:
            raise DistutilsFileError(
                "error listing files in '%s': %s" % (src, e.strerror)
            ) from e

    if not dry_run:
        mkpath(dst, verbose=verbose)

    outputs = []
    outputs_ignored = []

    for n in names:
        src_name = os.path.join(src, n)
        dst_name = os.path.join(dst, n)

        if n.startswith(".nfs"):
            # skip NFS rename files
            continue

        if preserve_symlinks and os.path.islink(src_name):
            link_dest = os.readlink(src_name)
            if verbose >= 1:
                log.info("linking %s -> %s", dst_name, link_dest)
            if not dry_run:
                os.symlink(link_dest, dst_name)
            outputs.append(dst_name)

        elif os.path.isdir(src_name):
            part_outputs, part_outputs_ignored = copytree(
                src_name,
                dst_name,
                ignore_pattern_if_file_exists,
                preserve_mode,
                preserve_times,
                preserve_symlinks,
                update,
                verbose=verbose,
                dry_run=dry_run,
            )
            outputs.extend(part_outputs)
            outputs_ignored.extend(part_outputs_ignored)

        elif not (
            fnmatch.fnmatch(dst_name, ignore_pattern_if_file_exists)
            and os.path.exists(dst_name)
        ):
            copy_file(
                src_name,
                dst_name,
                preserve_mode,
                preserve_times,
                update,
                verbose=verbose,
                dry_run=dry_run,
            )
            outputs.append(dst_name)

        else:
            outputs_ignored.append(dst_name)

    return outputs, outputs_ignored


def get_user_objects_from_module(
    module_str: str, custom_dirs: Optional[str] = None
):
    """
    Args:
        module_str: Module in the form of string, package.module.
        custom_dirs: custom directories to search from if `module_str` is not a
          part of imported modules.

    Returns: A list of objects present in the module `module_str`,
             None if module not found

    """
    module = locate(module_str)
    if module is not None and hasattr(module, "__all__"):
        return module.__all__  # type: ignore
    objects: List[str] = []
    if custom_dirs is not None:
        module_file = module_str.replace(".", "/") + ".py"
        for dir_ in custom_dirs:
            filepath = os.path.join(dir_, module_file)
            try:
                spec = import_util.spec_from_file_location(module_str, filepath)
                module = import_util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(module)  # type: ignore
                objects = module.__all__
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
        abs_dir = os.path.abspath(_dir)
        if not os.path.isabs(file):
            abs_file = os.path.join(abs_dir, file)
        abs_file = os.path.normpath(abs_file)
        if os.path.exists(abs_file):
            return abs_file
    return None


def get_module_path(module: str):
    module_spec = import_util.find_spec(module)
    return module_spec.origin if module_spec is not None else None


def get_top_level_dirs(path: Optional[str]):
    """
    Args:
        path: Path for which the directories at depth==1 are to be returned
    Returns:
        Directories at depth==1 for `path`.
    """
    if path is None or not os.path.exists(path):
        return []
    return [
        item
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
    ]


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
    prev_dir, curr_dir = None, (str(Path(path)), "")
    while prev_dir != curr_dir:
        prev_dir = curr_dir
        if curr_dir[-1].strip():
            path_split.append(curr_dir[-1])
        curr_dir = os.path.split(curr_dir[0])
    path_split += [""] if path.startswith("/") else []
    return path_split[::-1]


def validate_json_schema(input_filepath: str):
    """
    Validates the input json schema using validation meta-schema provided in
    `validation_filepath.json` according to the specification in
    `http://json-schema.org`.
    If the tested json is not valid, a `jsonschema.exceptions.ValidationError`
    is thrown.
    Args:
        input_filepath: Filepath of the json schema to be validated
    """
    validation_file_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "validation_schema.json")
    )
    with open(
        validation_file_path, "r", encoding="utf-8"
    ) as validation_json_file:
        validation_schema = json.loads(validation_json_file.read())
    with open(input_filepath, "r", encoding="utf-8") as input_json_file:
        input_schema = json.loads(input_json_file.read())
    jsonschema.Draft6Validator(validation_schema).validate(input_schema)


def get_python_version() -> Tuple[int, int]:
    """
    :return: Python major and minor version at runtime
    """
    version_info = sys.version_info
    return version_info[0], version_info[1]


def get_schema_from_ontology(
    imported_onto_file: Optional[str], delimiters: List[str]
) -> str:
    if imported_onto_file is None:
        raise FileNotFoundError
    with open(imported_onto_file, "r", encoding="utf-8") as imported_onto:
        regex = "|".join(map(re.escape, delimiters))
        reqd_line = imported_onto.readlines()[1]
        installed_json_file = list(filter(None, re.split(regex, reqd_line)))[0]
    return installed_json_file


def get_parent_path(file_path: str, level: int = 1):
    relative_path = os.path.join(file_path, *([os.pardir] * level))
    return os.path.normpath(relative_path)


def get_installed_forte_dir():
    init_path = get_module_path("forte")
    return get_parent_path(init_path, 2) if init_path is not None else None


def get_current_forte_dir():
    return get_parent_path(__file__, 4)


def get_generated_files_in_dir(path):
    def is_generated(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return len(lines) > 0 and lines[0] == f"# {AUTO_GEN_SIGNATURE}\n"

    ext_files = []
    for root, _, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if is_generated(path):
                ext_files.append(path)
    return ext_files
