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
    Module to automatically generate python ontology given json file
    Performs a preliminary check of dependencies
"""
import os
import json
import shutil
import warnings
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from distutils import dir_util
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Set, no_type_check
import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse

from forte.data.ontology import utils, top
from forte.data.ontology.code_generation_util import (
    PrimitiveProperty, CompositeProperty, ClassAttributeProperty,
    DefinitionItem, EntryWriter, Property)


# TODO: Causing error in sphinx - fix and uncomment. Current version displays
#  the line of code to the user, which is undesired.
# def format_warning(message, category, filename, lineno, _):
#     return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
#
#
# warnings.formatwarning = format_warning  # type: ignore

class DuplicatedAttributesWarning(UserWarning):
    pass


class DirectoryAlreadyPresentWarning(UserWarning):
    pass


class DuplicateEntriesWarning(UserWarning):
    pass


class ImportOntologyNotFoundException(ValueError):
    pass


class ImportOntologyAlreadyGeneratedException(ValueError):
    pass


class ParentEntryNotDeclaredException(ValueError):
    pass


class TypeNotDeclaredException(ValueError):
    pass


class NoDefaultClassAttributeException(ValueError):
    pass


class UnsupportedTypeException(ValueError):
    pass


class InvalidIdentifierException(ValueError):
    pass


def name_validation(name):
    parts = name.split('.')
    for part in parts:
        if not part.isidentifier():
            raise SyntaxError(f"'{part}' is not an valid identifier.")


def analyze_packages(ontology_spec_path: str, packages: Set[str]):
    r""" Analyze the package paths to make sure they are valid.

    Args:
        ontology_spec_path: The ontology specification path.
        packages: The list of packages.

    Returns: A list of the package paths, sorted by the package depth (
        deepest first).

    """
    package_len = []
    for p in packages:
        parts = p.split('.')
        package_len.append((len(parts), p))
        try:
            name_validation(p)
        except InvalidIdentifierException:
            logging.error(
                f"Error analyzing package name at file [{ontology_spec_path}]")
            raise

    return [p for (l, p) in sorted(package_len, reverse=True)]


def validate_entry(entry_name: str, sorted_packages: List[str]) -> str:
    for package_name in sorted_packages:
        if entry_name.startswith(package_name):
            matched_package = package_name
            break
    else:
        # None of the package name matches.
        raise InvalidIdentifierException(
            f"Entry name [{entry_name}] does not start with any predefined "
            f"packages, please define the packages in the ontology "
            f"specification. Or you can use the default prefix 'ft.onto'."
        )

    entry_splits = entry_name.split('.')

    if len(entry_splits) < 3:
        raise InvalidIdentifierException(
            f"We currently require each entry to contains at least 3 levels, "
            f"which corresponds to the directory name, the file (module) name,"
            f"the entry class name. There are only {len(entry_splits)}"
            f"levels in [{entry_name}].")
    return matched_package


class OntologyCodeGenerator:
    r"""
    Class to generate python ontology given ontology config in json format
    Salient Features -
        (1) Generates a class for each entry in the module corresponding to
        the defined entry package.
        (2) The entries of `forte.data.ontology.top`
        serve as ancestors of the user-defined entries.
        (2) Dependencies to other json configs through the json `imports`
    Example:
        >>> destination_dir = OntologyCodeGenerator().generate_ontology(
        'test/example_ontology_config.json')
    """
    AUTO_GEN_CONST = '***automatically_generated***'

    def __init__(self, json_dir_paths: Optional[List[str]] = None):
        """

        Args:
            json_dir_paths: Additional user provided paths to search the
            imported json configs from. By default paths provided in the json
            configs and the current working directory would be searched.
        """
        # The entries of the `self.base_ontology_module` serve as ancestors of
        # the user-defined entries.
        base_ontology_module: ModuleType = top

        # Builtin and local imports required in the generated python modules.
        self.required_imports: List[str] = [
            'typing',
            'ft.onto',
            'forte.data.data_pack',
            base_ontology_module.__name__]

        # Special comments to be added to disable checking.
        self.ignore_errors: List[str] = [
            f'# {self.AUTO_GEN_CONST}',
            '# flake8: noqa',
            '# mypy: ignore-errors',
            '# pylint: skip-file']

        self.core_base_keys = {"BaseLink": ["parent_type", "child_type"],
                               "GroupLink": ["member_type"]}

        # Mapping from entries parsed from the `base_ontology_module`
        # (default is `top.py`), to their `__init__` arguments.
        self.top_init_args: Dict[str, str]

        # Mapping from entries in `base_ontology_module` to their ancestors
        self.top_to_core_entries: Dict[str, Set[str]]

        self.top_init_args, _, self.top_to_core_entries = \
            self.initialize_top_entries(base_ontology_module)

        # Mapping from user-defined entries to their ancestor entry present in
        # `self.top_init_args`.
        self.user_to_base_entry: Dict[str, str] = {}

        # Mapping from reference string to corresponding full class name
        self.ref_to_full_name: Dict[str, str] = {}

        # Adjacency list to store the allowed types (in-built or user-defined),
        # and their attributes (if any) in order to validate the attribute
        # types.
        self.allowed_types_tree: Dict[str, Set] = {}
        for type_str in {*PrimitiveProperty.TYPES, *CompositeProperty.TYPES}:
            self.allowed_types_tree[type_str] = set()
            self.ref_to_full_name[type_str] = type_str

        # A temporary directory to save the generated file structure until the
        # generation is completed and verified.
        self.tempdir: str

        # Directories to be examined to find json files for user-defined config
        # imports.
        self.json_paths: List[str] = [] \
            if json_dir_paths is None else json_dir_paths

    @staticmethod
    @no_type_check
    def initialize_top_entries(base_ontology_module: ModuleType) \
            -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
        """
        Parses the file corresponding to `base_ontology_module` -
        (1) Imports the imports defined by the base file,
        (2) Imports the public API defined by by the base file in it's `__all__`
        attribute,
        (3) Extracts the name and inheritence of the class definitions and
        populates `self.top_to_core_entries`,
        (4) Extracts `__init__` arguments of class definitions and populates
        `self.top_init_args`
        (5) Includes type annotations for the `__init__` arguments.

        Args:
            base_ontology_module: File path of the module to be parsed.

        Returns:
            Mapping from a class name defined in `base_ontology_module`
        to the `__init__` arguments.
            Mapping from import names defined in `base_ontology_module` to
        import full name
            Mapping from import names defined in `base_ontology_module` to
        base names defined in `core.py`.
        """
        top_init_args = {}
        top_to_core_entries = {}
        tree = ast.parse(open(base_ontology_module.__file__, 'r').read())

        imports = {}

        for elem in tree.body:
            # Adding all the imports.
            if isinstance(elem, ast.Import):
                for import_ in elem.names:
                    as_name = import_.asname
                    import_name = import_.name if as_name is None else as_name
                    imports[import_name] = import_.name

            if isinstance(elem, ast.ImportFrom):
                for import_ in elem.names:
                    imports[import_.name] = f"{elem.module}.{import_.name}"

            # Adding all the module objects defined in `__all__` to imports.
            if isinstance(elem, ast.Assign) and len(elem.targets) > 0:
                if elem.targets[0].id == '__all__':
                    imports.update(
                        [(name.s,
                          f"{base_ontology_module.__name__}.{name.s}")
                         for name in elem.value.elts])

            # Adding `__init__` arguments for each class
            if isinstance(elem, ast.ClassDef):
                # Adding base names for each class
                elem_base_names = set()
                if elem.bases is not None and len(elem.bases) > 0:
                    for elem_base in elem.bases:
                        while isinstance(elem_base, ast.Subscript):
                            elem_base = elem_base.slice.value
                        elem_base_names.add(elem_base.id)
                init_func = None
                for func in elem.body:
                    if isinstance(func, ast.FunctionDef) and \
                            func.name == '__init__':
                        init_func = func
                        break
                if init_func is None:
                    warnings.warn(
                        f"No `__init__` function found in the class"
                        f" {elem.name} of the module "
                        f"{base_ontology_module}.")
                else:
                    # Assuming no variable args and keyword only args present in
                    # the base ontology module
                    for i, arg in enumerate(init_func.args.args):
                        # Parsing the nested list of arg annotations
                        if arg.annotation is not None:
                            arg_ann = arg.annotation
                            while isinstance(arg_ann, ast.Subscript):
                                module = arg_ann.value.id
                                if module is not None and module in imports:
                                    arg_ann.value.id = imports[module]
                                arg_ann = arg_ann.slice.value
                            module = arg_ann.id
                            if module is not None and module in imports:
                                arg_ann.id = imports[module]
                        init_func.args.args[i] = arg

                    # Unparsing the `__init__` args and normalising the string
                    args = ast_unparse.unparse(init_func.args).split(',', 1)
                    args_str = args[1].strip() \
                        .replace('\n', '').replace('  ', '')

                    top_to_core_entries[imports[elem.name]] = elem_base_names
                    top_init_args[imports[elem.name]] = args_str
        return top_init_args, imports, top_to_core_entries

    def generate_ontology(self, base_json_file_path: str,
                          destination_dir: Optional[str] = os.getcwd(),
                          is_dry_run: bool = False) -> Optional[str]:
        r"""Function to generate and save the python ontology code after reading
            ontology from the input json file.
            Args:
                base_json_file_path: The base ontology specification file.
                destination_dir: The folder in which config packages are to be
                generated. If not provided, current working directory is used.
                Ignored if `is_dry_run` is `True`.
                is_dry_run: if `True`, creates the ontology in the temporary
                directory, else, creates the ontology in the `destination_dir`.
            Returns:
                Directory path in which the modules are created: either one of
                the temporary directory or `destination_dir`.
        """
        self.tempdir = tempfile.mkdtemp()

        # Update the list of directories to be examined for imported configs
        self.json_paths.extend([
            os.path.dirname(os.path.realpath(base_json_file_path)),
            os.path.dirname(os.path.realpath('ft/onto')),
            os.getcwd()])

        # Adding the imported objects to the allowed types.
        for import_module in self.required_imports:
            for obj_str in utils.get_user_objects_from_module(import_module):
                full_obj_str = f"{import_module}.{obj_str}"
                self.allowed_types_tree[full_obj_str] = set()
                self.ref_to_full_name[obj_str] = full_obj_str

        # Generate ontology classes for the input json config and the configs
        # it is dependent upon.
        self.parse_ontology(base_json_file_path, destination_dir)

        # When everything is successfully completed, copy the contents of
        # `self.tempdir` to the provided folder.
        if not is_dry_run:

            generated_top_dirs = set(utils.get_top_level_dirs(self.tempdir))
            for existing_top_dir in utils.get_top_level_dirs(destination_dir):
                if existing_top_dir in generated_top_dirs:
                    warnings.warn(
                        f"The directory with the name "
                        f"{existing_top_dir} is already present in "
                        f"{destination_dir}. New files will be merge into the "
                        f"existing directory.", DirectoryAlreadyPresentWarning)

            dir_util.copy_tree(self.tempdir, destination_dir)

            return destination_dir
        return self.tempdir

    def parse_ontology(self, json_file_path: str,
                       destination_dir: str,
                       visited_paths: Optional[Dict[str, bool]] = None,
                       rec_visited_paths: Optional[Dict[str, bool]] = None) \
            -> List[str]:
        """
        Performs a topological traversal on the directed graph formed by the
        imported json configs. While processing each config, it first generates
        the classes corresponding to the entries of the imported configs, then
        imports the generated python classes to generate the classes
        corresponding to the entries of `json_file_path`.
        Args:
            json_file_path: The current json config to be processed.
            destination_dir:
            visited_paths: Keeps track of the json configs already processed.
            rec_visited_paths: Keeps track of the current recursion stack, to
            detect, and throw error if any cycles are present.
            with the base ontology config, else, False.

        Returns: Modules to be imported by the generated python files
        corresponding to the entries defined in json config imports.
        """
        # Initialize the visited dicts when the function is called for the
        # first time.
        if visited_paths is None:
            visited_paths = defaultdict(lambda: False)

        if rec_visited_paths is None:
            rec_visited_paths = defaultdict(lambda: False)

        visited_paths[json_file_path] = True
        rec_visited_paths[json_file_path] = True

        with open(json_file_path, 'r') as f:
            spec_str = f.read()

        spec_dict = json.loads(spec_str)

        # Extract imported json files and generate ontology for them.
        json_imports: List = spec_dict.get("import_paths", [])

        modules_to_import: List[str] = []

        for import_file in json_imports:
            import_json_file = utils.search_in_dirs(import_file,
                                                    self.json_paths)
            if import_json_file is None:
                raise ImportOntologyNotFoundException(
                    f"Ontology corresponding to {import_file} not "
                    f"found in the current directory or the "
                    f"directory of original json config.")
            if import_json_file in rec_visited_paths:
                raise ImportOntologyAlreadyGeneratedException(
                    f"Ontology corresponding to {import_json_file}"
                    f" already generated, cycles not permitted, "
                    f"aborting")
            elif import_json_file not in visited_paths:
                modules_to_import.extend(self.parse_ontology(
                    import_json_file, destination_dir,
                    visited_paths, rec_visited_paths))

        # Once the ontology for all the imported files is generated, generate
        # ontology of the current file
        modules_to_import = self.generate_from_schema(
            json_file_path, spec_dict, modules_to_import, destination_dir)
        rec_visited_paths[json_file_path] = False

        return modules_to_import

    def generate_from_schema(self,
                             ontology_spec_path: str,
                             schema: Dict, modules_to_import: List[str],
                             destination_dir: str) -> List[str]:
        r""" Generates ontology code for a parsed schema extracted from a
        json config. Appends entry code to the corresponding module. Creates a
        new module file if module is generated for the first time.

        Args:
            ontology_spec_path: The path of the ontology specification.
            schema: Ontology dictionary extracted from a json config.
            modules_to_import: Dependencies to be imported by generated modules.
            destination_dir:

        Returns:
            Modules to be imported by dependencies of the current ontology.
        """
        entry_definitions: List[Dict] = schema["definitions"]

        allowed_packages = set(schema.get("additional_prefixes", [])
                               + ["ft.onto"])
        sorted_prefixes = analyze_packages(ontology_spec_path, allowed_packages)

        new_modules_to_import = []
        for definition in entry_definitions:
            entry_name = definition["entry_name"]

            # Only prefixes that are actually used should be imported.
            matched_pkg = validate_entry(entry_name, sorted_prefixes)
            modules_to_import.append(matched_pkg)
            new_modules_to_import.append(matched_pkg)

            entry_splits = entry_name.split('.')
            filename, name = entry_splits[-2:]
            pkg = '.'.join(entry_splits[0: -2])

            self.ref_to_full_name[name] = entry_name
            self.ref_to_full_name[entry_name] = entry_name
            if entry_name in self.allowed_types_tree:
                warnings.warn(
                    f"Class {entry_name} already present in the "
                    f"ontology, will be overridden.", DuplicateEntriesWarning)
            self.allowed_types_tree[entry_name] = set()
            entry_item, properties = self.parse_entry(
                name, entry_name, definition)
            module_name: str = f"{pkg}.{filename}"
            class_name: str = f"{module_name}.{name}"

            try:
                # Creating entry directory and file in the tempdir if required.
                entry_pkg_dir = pkg.replace('.', '/')
                entry_dir: str = os.path.join(self.tempdir, entry_pkg_dir)
                entry_file: str = f'{os.path.join(entry_dir, filename)}.py'
                file_desc: str = schema.get('description', '')
                file_desc += '\n\nAutomatically generated file. ' \
                             'Do not change manually.'
                all_imports = self.required_imports + modules_to_import

                entry_writer = EntryWriter(
                    pkg, entry_item, entry_file, self.ignore_errors, file_desc,
                    all_imports
                )
                entry_writer.write(self.tempdir, destination_dir, filename)

            except ValueError:
                # TODO: I cannot find ValueError exception thrown in the
                #  code above, why this exception here?
                self.cleanup_generated_ontology(self.tempdir, is_forced=True)
                raise

            # Modules to be imported by the dependencies.
            new_modules_to_import.append(module_name)

            # Adding entry attributes to the allowed types for validation.
            for property_name in properties:
                if property_name in self.allowed_types_tree[class_name]:
                    warnings.warn(
                        f"Attribute type for the entry {class_name} and "
                        f"the attribute {property_name} already present in "
                        f"the ontology, will be overridden",
                        DuplicatedAttributesWarning
                    )
                self.allowed_types_tree[class_name].add(property_name)

        return new_modules_to_import

    def cleanup_generated_ontology(self, path, is_forced=False) -> \
            Tuple[bool, Optional[str]]:
        """
        Deletes the generated files and directories. Generated files are
        identified by the header `***automatically_generated***`. Generated
        directories are identified by `.generated` empty marker files. Moves
        the files to a timestamped folder inside `.deleted` folder located in
        the parent directory path by default.
        Args:
            path: Path of the directory to be searched for generated files.
            is_forced: Deletes the generated files and directories without
             moving them to `.deleted` directory. `False` by default.
        Returns:
            Whether `path` is empty after the cleanup operation is
        completed
            The timestamped `.deleted` directory path
        """
        path = os.path.abspath(path)

        rel_paths = dir_util.copy_tree(path, '', dry_run=1)
        rel_paths = [os.path.dirname(file) for file in rel_paths
                     if os.path.basename(file).startswith('.generated')]

        del_dir = None
        if not is_forced:
            curr_time_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
            del_dir = os.path.join(os.path.dirname(path), '.deleted',
                                   curr_time_str)
            for rel_path in rel_paths:
                joined_path = os.path.join(del_dir, rel_path)
                Path(joined_path).mkdir(parents=True, exist_ok=True)
        rel_paths += ['']
        return (self._cleanup_generated_ontology(path, '', del_dir, rel_paths),
                del_dir)

    def _cleanup_generated_ontology(self, outer_path, relative_path, delete_dir,
                                    allowed_relative_paths) -> bool:
        """
        Recursively deletes the generated files and the newly empty directories.
        Args:
            outer_path: Path of the directory to be searched for generated files
            relative_path: Path relative to outer path.
            delete_dir: Directory where the delete file structure has to be
            moved.
            allowed_relative_paths: Directory paths having `.generated` marker
            files.

        Returns: Whether `path` is empty after the cleanup operation is
        completed.
        """
        if os.path.dirname(relative_path) not in allowed_relative_paths:
            return False

        path = os.path.join(outer_path, relative_path)
        dst_dir = os.path.join(delete_dir, os.path.dirname(relative_path)) \
            if delete_dir is not None else None

        if os.path.isfile(path):
            # path is a file type
            # delete .generated marker files and automatically generated files
            is_empty = os.path.basename(path).startswith('.generated')
            if not is_empty and os.access(path, os.R_OK):
                with open(path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        if lines[0].startswith(f'# {self.AUTO_GEN_CONST}'):
                            is_empty = True
            if is_empty:
                if delete_dir is not None:
                    shutil.copy(path, dst_dir)
                os.unlink(path)
        else:
            # path is a directory type
            is_empty = True
            for child in os.listdir(path):
                child_rel_path = os.path.join(relative_path, child)
                if not self._cleanup_generated_ontology(outer_path,
                                                        child_rel_path,
                                                        delete_dir,
                                                        allowed_relative_paths):
                    is_empty = False
            if is_empty:
                if delete_dir is not None:
                    dir_util.copy_tree(path, dst_dir)
                os.rmdir(path)
        return is_empty

    def parse_entry(self, ref_name: str,
                    full_name: str,
                    schema: Dict) -> Tuple[DefinitionItem, List[str]]:
        """
        Args:
            ref_name: Reference name of the entry.
            full_name: Full class name of the entry.
            schema: Dictionary containing specifications for an entry.

        Returns: extracted entry information: entry package string, entry
        filename, entry class entry_name, generated entry code and entry
        attribute names.
        """
        name = full_name
        # reading the entry definition dictionary
        parent_entry: str = self.parse_type(schema["parent_entry"])

        properties: List[Dict] = schema.get("attributes", [])

        # validate if the entry parent is present in the tree
        if parent_entry not in self.allowed_types_tree:
            raise ParentEntryNotDeclaredException(
                f"Cannot add {name} to the ontology as "
                f"it's parent entry {parent_entry} is not present "
                f"in the ontology.")

        base_entry: str = self.get_and_set_base_entry(name, parent_entry)
        init_args: str = self.top_init_args[base_entry]
        core_bases: Set[str] = self.top_to_core_entries[base_entry]

        property_items, property_names = [], []
        for prop_schema in properties:
            property_names.append(prop_schema["name"])
            property_items.append(self.parse_property(name, prop_schema))

        class_attribute_names: List[str] = []
        if any([item == "BaseLink" for item in core_bases]):
            class_attribute_names = self.core_base_keys["BaseLink"]
        elif any([item == "BaseGroup" for item in core_bases]):
            class_attribute_names = self.core_base_keys["BaseGroup"]

        # TODO: Apply stricter checking on class attributes.
        # TODO: Test subtypes of Group.
        class_att_items: List[Property] = []

        for class_att in class_attribute_names:
            if class_att in schema:
                type_ = self.parse_type(schema[class_att])
                class_att_items.append(
                    ClassAttributeProperty(class_att, type_, ))

        entry_item = DefinitionItem(name=ref_name,
                                    class_type=parent_entry,
                                    init_args=init_args,
                                    properties=property_items,
                                    class_attributes=class_att_items,
                                    description=schema.get("description", None))

        return entry_item, property_names

    def parse_type(self, type_):
        return self.ref_to_full_name.get(type_, type_)

    def parse_property(self, entry_name: str, schema: Dict) -> Property:
        """
        Parses instance and class properties defined in an entry schema and
        checks for the constraints allowed by the ontology generation system.
        Args:
            entry_name: Normalized name for the entry as defined by the user.
            schema: Entry definition schema
        Returns: An object of class `code_generation_util.FileItem` containing
         the generated code.
        """
        name = schema["name"]
        type_str = schema["type"]
        type_ = self.parse_type(type_str)

        # schema type should be present in the validation tree
        if type_ not in self.allowed_types_tree:
            raise TypeNotDeclaredException(
                f"Attribute type '{type_}' for the entry "
                f"'{entry_name}' and the schema '{name}' not "
                f"declared in the ontology")

        desc = schema.get("description", None)
        default = schema.get("default", None)

        # TODO: Only supports array for now!
        # element type should be present in the validation tree
        if type_str in CompositeProperty.TYPES:
            if "item_type" not in schema:
                raise TypeNotDeclaredException(
                    f"Item type for the entry {entry_name} "
                    f"of the attribute {name} not declared")
            item_type = schema["item_type"]
            if item_type in CompositeProperty.TYPES:
                raise UnsupportedTypeException(
                    f"Item type {item_type} for the entry {entry_name} of the"
                    f"attribute {name} is a composite type, we do not support"
                    f"nested composite type.")
            item_type = self.parse_type(item_type)
            item_type_norm = item_type.replace('"', '')
            if not (item_type_norm in self.allowed_types_tree
                    or self.ref_to_full_name.get(item_type_norm, '')
                    in self.allowed_types_tree
                    or item_type_norm == entry_name):
                raise TypeNotDeclaredException(
                    f"Item type {item_type_norm} for the entry {entry_name} "
                    f"of the attribute {name} not declared in ontology")

            return CompositeProperty(
                name, type_, item_type, description=desc,
                default=default)

        return PrimitiveProperty(name, type_, description=desc,
                                 default=default)

    def parse_attribute(self, entry_name, schema):
        name = schema["name"]
        type_str = schema.get("type", None)
        type_ = '' if type_str is None else self.parse_type(type_str)

        desc = schema.get("description", None)

        if "default" not in schema:
            raise NoDefaultClassAttributeException(
                f"No default value present for the class attribute"
                f" {name} for the entry {entry_name}.")

        # if default is of the type "type" which is already seen, parse_type
        default = self.parse_type(schema.get("default"))
        return ClassAttributeProperty(name, type_, desc, default)

    def get_and_set_base_entry(self, entry_name: str, parent_entry: str) \
            -> str:
        """
        Function to return `base_entry` which is the entry on which the
         arguments would be based and populates `self.user_to_base_entry`
        Args:
            entry_name: the entry name for which the base_entry is
            to be returned.
            parent_entry: parent of the entry name Note that the `base_entry` of
            the `entry_name` is same as the base entry of the `parent_entry`.

        Returns:
            `base_entry` for the entry `entry_name`. Note that base entry can
            only be one of - `["Link", "Annotation", "Group"]`

        Example:
            If the subclass structure is -
            `Word` inherits `Token` inherits `Annotation`.
            The base_entry for both `Word` and `Token` should be `Annotation`.
        """
        if parent_entry in self.top_init_args:
            base_entry: str = parent_entry
        else:
            base_entry = self.user_to_base_entry[parent_entry]
        self.user_to_base_entry[entry_name] = base_entry
        return base_entry
