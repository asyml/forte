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
import copy
import json
import logging
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime
from distutils import dir_util
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Set, no_type_check, Any

import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse

from forte.data.ontology import top, utils
from forte.data.ontology.code_generation_exceptions import \
    DirectoryAlreadyPresentWarning, DuplicateEntriesWarning, OntologySpecError, \
    ImportOntologyNotFoundException, ImportOntologyAlreadyGeneratedException, \
    ParentEntryNotDeclaredException, TypeNotDeclaredException, \
    UnsupportedTypeException, InvalidIdentifierException, \
    DuplicatedAttributesWarning, ParentEntryNotSupportedException
from forte.data.ontology.code_generation_objects import (
    PrimitiveProperty, CompositeProperty, ClassTypeDefinition,
    DefinitionItem, Property, ImportManagerPool,
    EntryName, ModuleWriterPool, ImportManager, DictProperty)
# Builtin and local imports required in the generated python modules.
from forte.data.ontology.ontology_code_const import REQUIRED_IMPORTS, \
    DEFAULT_CONSTRAINTS_KEYS, AUTO_GEN_SIGNATURE, DEFAULT_PREFIX, \
    SchemaKeywords, file_header, hardcoded_pack_map, PRIMITIVE_SUPPORTED, \
    SINGLE_COMPOSITES, COMPLEX_COMPOSITES


# TODO: Causing error in sphinx - fix and uncomment. Current version displays
#  the line of code to the user, which is undesired.
# def format_warning(message, category, filename, lineno, _):
#     return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
#
#
# warnings.formatwarning = format_warning  # type: ignore


# Special comments to be added to disable checking.


def name_validation(name):
    parts = name.split('.')
    for part in parts:
        if not part.isidentifier():
            raise SyntaxError(f"'{part}' is not an valid identifier.")


def analyze_packages(packages: Set[str]):
    r""" Analyze the package paths to make sure they are valid.

    Args:
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
                f"Error analyzing package name: '{p}' is "
                f"not a valid package name")
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


def as_init_str(init_args):
    """
    Create the __init__ string by using unparse in ast.
    Args:
        init_args: The ast args object of the init arguments.

    Returns:

    """
    # Unparsing the `__init__` args and normalising the string
    args = ast_unparse.unparse(init_args).strip().split(',', 1)
    return args[1].strip().replace('  ', '')


def is_composite_type(item_type: str):
    return item_type in SINGLE_COMPOSITES or item_type == 'Dict'


def valid_composite_key(item_type: str):
    return item_type == 'int' or item_type == 'str'


class OntologyCodeGenerator:
    r"""Class to generate python ontology given ontology config in json format
    Salient Features -
        (1) Generates a class for each entry in the module corresponding to
        the defined entry package.
        (2) The entries of `forte.data.ontology.top`
        serve as ancestors of the user-defined entries.
        (2) Dependencies to other json configs through the json `imports`
    Example:
        >>> destination_dir = OntologyCodeGenerator().generate(
        'test/example_ontology_config.json')
    """

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
        self.required_imports: List[str] = REQUIRED_IMPORTS
        self.required_imports.append(base_ontology_module.__name__)

        # A collection of import managers: each manager is responsible for
        # controlling the imports of one module. The key of the collection is
        # the module's full name, e.g. ft.onto.base_ontology
        self.import_managers: ImportManagerPool = ImportManagerPool()

        # Store information to write each modules.
        self.module_writers: ModuleWriterPool = ModuleWriterPool(
            self.import_managers)

        # Mapping from entries parsed from the `base_ontology_module`
        # (default is `forte.data.ontology.top.py`), to their
        # `__init__` arguments.
        # self.top_init_args_strs: Dict[str, str] = {}
        self.root_base_entrys: Set[str] = set()

        # Map from the full class name, to the list contains objects of
        # <typed_ast._ast3.arg>, which are the init arguments.
        self.top_init_args: Dict[str, Any] = {}

        # Mapping from user extendable entries to their ancestors
        self.top_to_core_entries: Dict[str, Set[str]] = {}

        # Mapping from user-defined entries to a set of base ancestor entries.
        # these base entries are the top level entries where user can
        # initialize, so they wil be part of the generated class's __init__
        self.base_entry_lookup: Dict[str, str] = {}

        # Populate the two dictionaries above.
        # TODO: Handle the imports from root, such as typing.
        self.initialize_top_entries(base_ontology_module)

        # A few basic type to support.
        self.import_managers.root.add_object_to_import('typing.Optional')

        for type_class in COMPLEX_COMPOSITES.values():
            self.import_managers.root.add_object_to_import(type_class)

        for type_class in SINGLE_COMPOSITES.values():
            self.import_managers.root.add_object_to_import(type_class)

        # Mapping from the full class name to the ref string to be used here.
        # self.full_ref_to_import: Dict[str, str] = {}

        # Adjacency list to store the allowed types (in-built or user-defined),
        # and their attributes (if any) in order to validate the attribute
        # types.
        self.allowed_types_tree: Dict[str, Set] = {}
        for type_str in {*PRIMITIVE_SUPPORTED}:
            self.allowed_types_tree[type_str] = set()
            # self.import_manager.add_object_to_import(type_str, True)

        # for type_name, type_str in CompositeProperty.TYPES.items():
        #     self.allowed_types_tree[type_str] = set()
        # self.import_manager.add_object_to_import(type_str, False)

        # Directories to be examined to find json files for user-defined config
        # imports.
        self.json_paths: List[str] = [] \
            if json_dir_paths is None else json_dir_paths

    @no_type_check
    def initialize_top_entries(self, base_ontology_module: ModuleType):
        """
        Parses the file corresponding to `base_ontology_module` -
        (1) Imports the imports defined by the base file,
        (2) Imports the public API defined by by the base file in it's `__all__`
        attribute,
        (3) Extracts the name and inheritance of the class definitions and
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
        import full name.
            Mapping from import names defined in `base_ontology_module` to
        base names defined in `core.py`.
        """
        tree = ast.parse(open(base_ontology_module.__file__, 'r').read())
        base_module_name = base_ontology_module.__name__

        full_names = {}
        root_manager = self.import_managers.root

        for elem in tree.body:
            # Adding all the imports.
            if isinstance(elem, ast.Import):
                for import_ in elem.names:
                    as_name = import_.asname
                    import_name = import_.name if as_name is None else as_name
                    full_names[import_name] = import_.name
                    root_manager.add_object_to_import(import_.name)

            if isinstance(elem, ast.ImportFrom):
                for import_ in elem.names:
                    full_names[import_.name] = f"{elem.module}.{import_.name}"
                    full_name = f"{elem.module}.{import_.name}"
                    root_manager.add_object_to_import(full_name)

            # Adding all the module objects defined in `__all__` to imports.
            if isinstance(elem, ast.Assign) and len(elem.targets) > 0:
                if elem.targets[0].id == '__all__':
                    full_names.update(
                        [(name.s,
                          f"{base_ontology_module.__name__}.{name.s}")
                         for name in elem.value.elts])

                    for name in elem.value.elts:
                        full_class_name = f"{base_module_name}.{name.s}"
                        root_manager.add_object_to_import(full_class_name)

            # Adding `__init__` arguments for each class
            if isinstance(elem, ast.ClassDef):

                # Adding base names for each class
                elem_base_names = set()
                if elem.bases is not None and len(elem.bases) > 0:
                    for elem_base in elem.bases:
                        while isinstance(elem_base, ast.Subscript):
                            # TODO: Doesn't handle typed class well.
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
                                if module is not None and module in full_names:
                                    arg_ann.value.id = full_names[module]
                                arg_ann = arg_ann.slice.value
                            module = arg_ann.id
                            if module is not None and module in full_names:
                                pack_type = hardcoded_pack_map(
                                    full_names[elem.name])
                                if pack_type is None:
                                    # Set the annotation id to the full name.
                                    arg_ann.id = full_names[module]
                                else:
                                    arg_ann.id = pack_type

                                root_manager.add_object_to_import(arg_ann.id)

                        # No need to replace this, the object is modified in
                        # place.
                        # init_func.args.args[i] = arg

                    # Unparsing the `__init__` args and normalising the string
                    args = ast_unparse.unparse(init_func.args).split(',', 1)
                    args_str = args[1].strip().replace(
                        '\n', '').replace('  ', '')

                    full_ele_name = full_names[elem.name]
                    self.top_to_core_entries[full_ele_name] = elem_base_names
                    self.base_entry_lookup[full_ele_name] = full_ele_name
                    self.top_init_args[full_ele_name] = init_func.args

    def generate(self, spec_path: str,
                 destination_dir: Optional[str] = os.getcwd(),
                 is_dry_run: bool = False) -> Optional[str]:
        r"""Function to generate and save the python ontology code after reading
            ontology from the input json file. This is the main entry point to
            the class.

            Args:
                spec_path: The input ontology specification file, which should
                    be a json file.
                destination_dir: The folder in which config packages are to be
                    generated. If not provided, current working directory is
                    used. Ignored if `is_dry_run` is `True`.
                is_dry_run: if `True`, creates the ontology in the temporary
                    directory, else, creates the ontology in the
                    `destination_dir`.

            Returns:
                Directory path in which the modules are created: either one of
                the temporary directory or `destination_dir`.
        """
        # Update the list of directories to be examined for imported configs
        self.json_paths.extend([
            os.path.dirname(os.path.realpath(spec_path)),
            os.path.dirname(os.path.realpath('ft/onto')),
            os.getcwd()])

        # TODO: validate the JSON paths here.

        # TODO: This section does the required imports, maybe useless.
        # Adding the imported objects to the allowed types.
        for import_module in self.required_imports:
            for obj_str in utils.get_user_objects_from_module(import_module):
                full_obj_str = f"{import_module}.{obj_str}"
                self.allowed_types_tree[full_obj_str] = set()
                # self.import_manager.add_object_to_import(full_obj_str, False)
                # self.full_ref_to_import[obj_str] = full_obj_str

        # Generate ontology classes for the input json config and the configs
        # it is dependent upon.
        try:
            self.parse_ontology_spec(spec_path, destination_dir)
        except OntologySpecError:
            logging.error(f"Error at parsing [{spec_path}]")
            raise

        # Now generate all data.

        # A temporary directory to save the generated file structure until the
        # generation is completed and verified.
        tempdir = tempfile.mkdtemp()

        print('*********** finish parsing start writing')
        print('working on ', spec_path)
        for writer in self.module_writers.writers():
            print('writing ', writer.module_name)
            writer.write(tempdir, destination_dir)
            print('Done writing.')

        # When everything is successfully completed, copy the contents of
        # `self.tempdir` to the provided folder.
        if not is_dry_run:
            generated_top_dirs = set(utils.get_top_level_dirs(tempdir))
            for existing_top_dir in utils.get_top_level_dirs(destination_dir):
                if existing_top_dir in generated_top_dirs:
                    warnings.warn(
                        f"The directory with the name "
                        f"{existing_top_dir} is already present in "
                        f"{destination_dir}. New files will be merge into the "
                        f"existing directory.", DirectoryAlreadyPresentWarning)

            dir_util.copy_tree(tempdir, destination_dir)

            return destination_dir
        return tempdir

    def parse_ontology_spec(self, json_file_path: str,
                            destination_dir: str,
                            visited_paths: Optional[Dict[str, bool]] = None,
                            rec_visited_paths: Optional[Dict[str, bool]] = None
                            ) -> List[str]:
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

        # Load the ontology specification.
        with open(json_file_path, 'r') as f:
            spec_dict = json.load(f)

        # Extract imported json files and generate ontology for them.
        json_imports: List[str] = spec_dict.get("import_paths", [])

        # Store the modules contained in the specifications.
        spec_importable_modules: List[str] = []

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
                spec_importable_modules.extend(self.parse_ontology_spec(
                    import_json_file, destination_dir,
                    visited_paths, rec_visited_paths))

        # The modules from the imported specs will be added to the manager.
        for spec_module in spec_importable_modules:
            self.import_managers.root.add_object_to_import(spec_module)

        # Once the ontology for all the imported files is generated, generate
        # ontology of the current file.
        self.parse_schema(spec_dict, spec_importable_modules)

        rec_visited_paths[json_file_path] = False

        return spec_importable_modules

    def parse_schema(self, schema: Dict, modules_to_import: List[str]):
        r""" Generates ontology code for a parsed schema extracted from a
        json config. Appends entry code to the corresponding module. Creates a
        new module file if module is generated for the first time.

        Args:
            schema: Ontology dictionary extracted from a json config.

        Returns:
            Modules to be imported by dependencies of the current ontology.
        """
        entry_definitions: List[Dict] = schema[SchemaKeywords.definitions]

        allowed_packages = set(
            schema.get(SchemaKeywords.prefixes, []) + [DEFAULT_PREFIX])
        sorted_prefixes = analyze_packages(allowed_packages)

        file_desc = file_header(
            schema.get(SchemaKeywords.description, ''),
            schema.get(SchemaKeywords.ontology_name, "")
        )

        # new_modules_to_import = []
        for definition in entry_definitions:
            raw_entry_name = definition[SchemaKeywords.entry_name]

            # Only prefixes that are actually used should be imported.
            matched_pkg = validate_entry(raw_entry_name, sorted_prefixes)
            # new_modules_to_import.append(matched_pkg)

            if raw_entry_name in self.allowed_types_tree:
                warnings.warn(
                    f"Class {raw_entry_name} already present in the "
                    f"ontology, will be overridden.", DuplicateEntriesWarning)
            self.allowed_types_tree[raw_entry_name] = set()

            # Get various name of this entry.
            en = EntryName(raw_entry_name)
            module_writer = self.module_writers.get(en.module_name)
            module_writer.set_description(file_desc)

            # Add the entry definition to the import managers.
            self.import_managers.get(en.module_name).add_object_to_import(
                raw_entry_name)
            # Add the module writer.

            entry_item, properties = self.parse_entry(en, definition)

            # Add the entry item to the writer.
            module_writer.add_entry(en, entry_item)

            # Modules to be imported by the dependencies.
            modules_to_import.append(en.class_name)

            # Adding entry attributes to the allowed types for validation.
            for property_name in properties:
                if property_name in self.allowed_types_tree[en.class_name]:
                    warnings.warn(
                        f"Attribute type for the entry {en.class_name} "
                        f"and the attribute {property_name} already present in "
                        f"the ontology, will be overridden",
                        DuplicatedAttributesWarning
                    )
                self.allowed_types_tree[en.class_name].add(
                    property_name)

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
                        if lines[0].startswith(f'# {AUTO_GEN_SIGNATURE}'):
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

    def replace_annotation(self, entry_name: EntryName, func_args):
        this_manager = self.import_managers.get(entry_name.module_name)
        for i, arg in enumerate(func_args.args):
            if arg.annotation is not None:
                arg_ann = arg.annotation
                while isinstance(arg_ann, ast.Subscript):
                    # Handling the type name for cases like Optional[X]
                    if this_manager.is_imported(arg_ann.value.id):
                        arg_ann.value.id = this_manager.get_name_to_use(
                            arg_ann.value.id)
                    arg_ann = arg_ann.slice.value

                if this_manager.is_imported(arg_ann.id):
                    # Handling the type name for arguments.
                    arg_ann.id = this_manager.get_name_to_use(arg_ann.id)

    def construct_init(self, entry_name: EntryName, base_entry: str):
        base_init_args = self.top_init_args[base_entry]
        custom_init_args = copy.deepcopy(base_init_args)
        self.replace_annotation(entry_name, custom_init_args)
        custom_init_args_str = as_init_str(custom_init_args)

        return custom_init_args_str

    def parse_entry(self, entry_name: EntryName,
                    schema: Dict) -> Tuple[DefinitionItem, List[str]]:
        """
        Args:
            entry_name: Object holds various name form of the entry.
            schema: Dictionary containing specifications for an entry.

        Returns: extracted entry information: entry package string, entry
        filename, entry class entry_name, generated entry code and entry
        attribute names.
        """
        # Determine the parent entry of this entry.
        parent_entry: str = schema[SchemaKeywords.parent_entry]
        base_entry: str = self.find_base_entry(entry_name.class_name,
                                               parent_entry)

        if base_entry is None or base_entry not in self.top_init_args:
            raise ParentEntryNotSupportedException(
                f"Cannot add {entry_name.class_name} to the ontology as "
                f"it's parent entry {parent_entry} is not supported. This is "
                f"likely that the entries are not inheriting the allowed types."
            )

        # Take the property definitions of this entry.
        properties: List[Dict] = schema.get(SchemaKeywords.attributes, [])

        this_manager = self.import_managers.get(entry_name.module_name)

        # Validate if the entry parent is presented.
        if not this_manager.is_known_name(parent_entry):
            raise ParentEntryNotDeclaredException(
                f"Cannot add {entry_name.class_name} to the ontology as "
                f"it's parent entry {parent_entry} is not present "
                f"in the ontology.")

        parent_entry_use_name = parent_entry
        if this_manager.is_imported(parent_entry):
            parent_entry_use_name = this_manager.get_name_to_use(parent_entry)

        property_items, property_names = [], []
        for prop_schema in properties:
            property_names.append(prop_schema["name"])
            property_items.append(
                self.parse_property(entry_name, prop_schema))

        # For special classes that requires a constraint.
        core_bases: Set[str] = self.top_to_core_entries[base_entry]
        entry_constraint_keys: Dict[str, str] = {}
        if any([item == "BaseLink" for item in core_bases]):
            entry_constraint_keys = DEFAULT_CONSTRAINTS_KEYS["BaseLink"]
        elif any([item == "BaseGroup" for item in core_bases]):
            entry_constraint_keys = DEFAULT_CONSTRAINTS_KEYS["BaseGroup"]

        # TODO: Apply stricter checking on class attributes.
        class_att_items: List[ClassTypeDefinition] = []

        for constraint_key, constraint_code in entry_constraint_keys.items():
            if constraint_key in schema:
                constraint_type_ = schema[constraint_key]
                constraint_type_use_name = this_manager.get_name_to_use(
                    constraint_type_)
                class_att_items.append(
                    ClassTypeDefinition(constraint_code,
                                        constraint_type_use_name))

        custom_init_arg_str: str = self.construct_init(entry_name, base_entry)

        entry_item = DefinitionItem(
            name=entry_name.name,
            class_type=parent_entry_use_name,
            init_args=custom_init_arg_str,
            properties=property_items,
            class_attributes=class_att_items,
            description=schema.get(SchemaKeywords.description, None))

        return entry_item, property_names

    def parse_dict(
            self, manager: ImportManager, schema: Dict, entry_name: EntryName,
            att_name: str, att_type: str, desc: str):
        if (SchemaKeywords.dict_key_type not in schema
                or SchemaKeywords.dict_value_type not in schema):
            raise TypeNotDeclaredException(
                f"Item type for the entry {entry_name.name} "
                f"of the attribute {att_name} not declared. This attribute is "
                f"a composite type: {att_type}, it should have a "
                f"{SchemaKeywords.element_type} and "
                f"{SchemaKeywords.dict_value_type}.")

        key_type = schema[SchemaKeywords.dict_key_type]
        if not valid_composite_key(key_type):
            raise UnsupportedTypeException(
                f"Key type {key_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is not supported, we only support a "
                f"limited set of keys.")

        value_type = schema[SchemaKeywords.dict_value_type]
        if is_composite_type(value_type):
            # Case of nested.
            raise UnsupportedTypeException(
                f"Item type {value_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is a composite type, we do not support "
                f"nested composite type.")

        if not manager.is_known_name(value_type):
            # Case of unknown.
            raise TypeNotDeclaredException(
                f"Item type {value_type} for the entry "
                f"{entry_name.name} of the attribute {att_name} "
                f"not declared in ontology.")

        # Make sure the import of these related types are handled.
        full_type = COMPLEX_COMPOSITES['Dict']
        manager.add_object_to_import(full_type)
        manager.add_object_to_import(value_type)

        self_ref = entry_name.class_name == value_type

        default_val = None

        if att_type == 'List':
            default_val = []
        elif att_type == 'Set':
            default_val = set()

        return DictProperty(
            manager, att_name, key_type, value_type, description=desc,
            default_val=default_val, self_ref=self_ref)

    def parse_single_composite(
            self, manager: ImportManager, schema: Dict, entry_name: EntryName,
            att_name: str, att_type: str, desc: str) -> CompositeProperty:
        if SchemaKeywords.element_type not in schema:
            raise TypeNotDeclaredException(
                f"Item type for the entry {entry_name.name} "
                f"of the attribute {att_name} not declared. This attribute is "
                f"a composite type: {att_type}, it should have a "
                f"{SchemaKeywords.element_type}.")

        item_type = schema[SchemaKeywords.element_type]
        if is_composite_type(item_type):
            # Case of nested.
            raise UnsupportedTypeException(
                f"Item type {item_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is a composite type, we do not support "
                f"nested composite type.")

        if not manager.is_known_name(item_type):
            # Case of unknown.
            raise TypeNotDeclaredException(
                f"Item type {item_type} for the entry "
                f"{entry_name.name} of the attribute {att_name} "
                f"not declared in ontology.")

        full_type = SINGLE_COMPOSITES[att_type]

        # Make sure the import of these related types are handled.
        manager.add_object_to_import(full_type)
        manager.add_object_to_import(item_type)

        self_ref = entry_name.class_name == item_type

        default_val = None

        if att_type == 'List':
            default_val = []
        elif att_type == 'Set':
            default_val = set()

        return CompositeProperty(
            manager, att_name, full_type, item_type, description=desc,
            default_val=default_val, self_ref=self_ref)

    def parse_property(self, entry_name: EntryName, schema: Dict) -> Property:
        """
        Parses instance and class properties defined in an entry schema and
        checks for the constraints allowed by the ontology generation system.
        Args:
            entry_name: Entry Name object that contains various form of the
            entry's name.
            schema: Entry definition schema
        Returns: An object of class `code_generation_util.FileItem` containing
         the generated code.
        """
        att_name = schema[SchemaKeywords.attribute_name]
        att_type = schema[SchemaKeywords.attribute_type]

        manager: ImportManager = self.import_managers.get(
            entry_name.module_name)

        # schema type should be present in the validation tree
        if not manager.is_known_name(att_type):
            raise TypeNotDeclaredException(
                f"Attribute type '{att_type}' for the entry "
                f"'{entry_name.name}' of the attribute '{att_name}' not "
                f"declared in the ontology")

        desc = schema.get(SchemaKeywords.description, None)
        default_val = schema.get(SchemaKeywords.default_value, None)

        # TODO: Only supports array for now!
        # element type should be present in the validation tree
        if att_type in SINGLE_COMPOSITES:
            return self.parse_single_composite(
                manager, schema, entry_name, att_name, att_type, desc)
        elif att_type == 'Dict':
            return self.parse_dict(
                manager, schema, entry_name, att_name, att_type, desc)
        else:
            return PrimitiveProperty(
                manager, att_name, att_type, description=desc,
                default_val=default_val)

    def find_base_entry(self, this_entry: str, parent_entry: str) -> str:
        """ Find the `base_entry`. As a side effect, it will populate the
        internal state `self.base_entry_lookup`. The base will be one of the
        predefined base entry group in the top ontology, which are:
         ["Link", "Annotation", "Group", "Generic", "MultiPackLink",
         "MultiPackGroup", "MultiPackGeneric"]

        Args:
            this_entry: the entry name for which the base_entry is to be
               returned.
            parent_entry: parent of the `this_entry`. Note that the `base_entry`
             of the `this_entry` is same as the base entry of the
             `parent_entry`.

        Returns:
            `base_entry` for the entry `this_entry`.

        Example:
            If the subclass structure is -
            `Word` inherits `Token` inherits `Annotation`.
            The base_entry for both `Word` and `Token` should be `Annotation`.
        """
        if parent_entry in self.top_init_args:
            # The top init args contains the objects in the top.py.
            base_entry: str = parent_entry
        else:
            base_entry = self.base_entry_lookup.get(parent_entry, None)

        if base_entry is not None:
            self.base_entry_lookup[this_entry] = base_entry

        return base_entry
