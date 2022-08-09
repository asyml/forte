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

import jsonschema
import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse
from numpy import ndarray

from forte.data.ontology import top, utils
from forte.data.ontology.code_generation_exceptions import (
    DuplicateEntriesWarning,
    OntologySpecError,
    OntologyAlreadyGeneratedException,
    ParentEntryNotDeclaredException,
    TypeNotDeclaredException,
    UnsupportedTypeException,
    InvalidIdentifierException,
    DuplicatedAttributesWarning,
    ParentEntryNotSupportedException,
    OntologySpecValidationError,
    OntologySourceNotFoundException,
)
from forte.data.ontology.code_generation_objects import (
    NdArrayProperty,
    NonCompositeProperty,
    ListProperty,
    ClassTypeDefinition,
    EntryDefinition,
    Property,
    ImportManagerPool,
    EntryName,
    ModuleWriterPool,
    ImportManager,
    DictProperty,
    EntryTree,
)

# Builtin and local imports required in the generated python modules.
from forte.data.ontology.ontology_code_const import (
    REQUIRED_IMPORTS,
    DEFAULT_CONSTRAINTS_KEYS,
    AUTO_GEN_SIGNATURE,
    DEFAULT_PREFIX,
    SchemaKeywords,
    file_header,
    NON_COMPOSITES,
    COMPOSITES,
    ALL_INBUILT_TYPES,
    TOP_MOST_MODULE_NAME,
    PACK_TYPE_CLASS_NAME,
    hardcoded_pack_map,
    AUTO_GEN_FILENAME,
    AUTO_DEL_FILENAME,
    RESERVED_ATTRIBUTE_NAMES,
)
from forte.utils.utils_io import get_resource


def name_validation(name):
    parts = name.split(".")
    for part in parts:
        if not part.isidentifier():
            raise SyntaxError(f"'{part}' is not an valid identifier.")


def analyze_packages(packages: Set[str]):
    r"""Analyze the package paths to make sure they are valid.

    Args:
        packages: The list of packages.

    Returns: A list of the package paths, sorted by the package depth (
        deepest first).

    """
    package_len = []
    for p in packages:
        parts = p.split(".")
        package_len.append((len(parts), p))
        try:
            name_validation(p)
        except InvalidIdentifierException:
            logging.error(
                "Error analyzing package name: '%s' is "
                "not a valid package name",
                p,
            )
            raise

    return [p for (l, p) in sorted(package_len, reverse=True)]


def validate_entry(
    entry_name: str, sorted_packages: List[str], lenient_prefix=False
):
    """
    Validate if this entry name can be used. It currently checks for:
      1) If the package name is defined. (This can be turn off by setting
        `lenient_prefix` to True.
      2) If the entry name have at least 3 segments.

    Args:
        entry_name: The name to be validated.
        sorted_packages: The package names that are allowed.
        lenient_prefix: Whether we enforce that the entry must follow the
          pre-defined package name.

    Returns:

    """
    if not lenient_prefix:
        for package_name in sorted_packages:
            if entry_name.startswith(package_name):
                break
        else:
            # None of the package name matches.
            raise InvalidIdentifierException(
                f"Entry name [{entry_name}] does not start with any predefined "
                f"packages, please define the packages by using "
                f"`additional_prefixes` in the ontology. Or you can use the "
                f"default prefix 'ft.onto'."
            )

    entry_splits = entry_name.split(".")

    for e in entry_splits:
        if not e.isidentifier():
            raise InvalidIdentifierException(
                f"The entry name segment {e} is not a valid python identifier."
            )

    if len(entry_splits) < 3:
        raise InvalidIdentifierException(
            f"We currently require each entry to contains at least 3 levels, "
            f"which corresponds to the directory name, the file (module) name,"
            f"the entry class name. There are only {len(entry_splits)}"
            f"levels in [{entry_name}]."
        )


def as_init_str(init_args):
    """
    Create the __init__ string by using unparse in ast.
    Args:
        init_args: The ast args object of the init arguments.

    Returns:

    """
    # Unparsing the `__init__` args and normalising the string
    args = ast_unparse.unparse(init_args).strip().split(",", 1)
    return args[1].strip().replace("  ", "")


def is_composite_type(item_type: str):
    return item_type in COMPOSITES


def valid_composite_key(item_type: str):
    return item_type in ("int", "str")


class OntologyCodeGenerator:
    r"""Class to generate python ontology given ontology config in json format
    Salient Features -
        (1) Generates a class for each entry in the module corresponding to
        the defined entry package.
        (2) The entries of `forte.data.ontology.top`
        serve as ancestors of the user-defined entries.
        (2) Dependencies to other json configs through the json `imports`
    Example:

        .. code-block:: python

            destination_dir = OntologyCodeGenerator().generate(
                ... 'forte/ontology_specs/base_ontology.json')

    """

    def __init__(
        self, import_dirs: Optional[List[str]] = None, generate_all=False
    ):
        """
        Args:
            import_dirs: Additional user provided paths to search the
                imported json configs or installed ontologies from. By default
                paths provided in the imports, current working directory and
                the path where forte is installed (if it is) would be searched.
            generate_all: whether to generate all the packages even if some are
                already generated.
        """
        # The entries of the `self.top_ontology_module` serve as ancestors of
        # the user-defined entries.
        top_ontology_module: ModuleType = top

        # Builtin and local imports required in the generated python modules.
        self.required_imports: List[str] = REQUIRED_IMPORTS
        self.required_imports.append(top_ontology_module.__name__)

        # A collection of import managers: each manager is responsible for
        # controlling the imports of one module. The key of the collection is
        # the module's full name, e.g. ft.onto.base_ontology
        self.import_managers: ImportManagerPool = ImportManagerPool()

        # Store information to write each modules.
        self.module_writers: ModuleWriterPool = ModuleWriterPool(
            self.import_managers
        )

        # Mapping from entries parsed from the `top_ontology_module`
        # (default is `forte.data.ontology.top.py`), to their
        # `__init__` arguments.
        # self.top_init_args_strs: Dict[str, str] = {}
        self.root_base_entries: Set[str] = set()

        # Map from the full class name, to the list contains objects of
        # <typed_ast._ast3.arg>, which are the init arguments.
        self.top_init_args: Dict[str, Any] = {}

        # Mapping from user extendable entries to their ancestors
        self.top_to_core_entries: Dict[str, Set[str]] = {}

        # Mapping from user-defined entries to a set of base ancestor entries.
        # these base entries are the top level entries where user can
        # initialize, so they wil be part of the generated class's __init__
        self.base_entry_lookup: Dict[str, str] = {}

        # Populate the two dictionaries above. And make the classes in the base
        # ontology aware to the root manager.
        self.initialize_top_entries(
            self.import_managers.root, top_ontology_module
        )

        # A few pre-requesite type to support.
        self.import_managers.add_default_import("dataclasses.dataclass")
        self.import_managers.root.add_object_to_import("typing.Optional")

        for type_class in NON_COMPOSITES.values():
            self.import_managers.root.add_object_to_import(type_class)

        # Adjacency list to store the allowed types (in-built or user-defined),
        # and their attributes (if any) in order to validate the attribute
        # types.
        self.allowed_types_tree: Dict[str, Set] = {}

        for type_str in ALL_INBUILT_TYPES:
            self.allowed_types_tree[type_str] = set()

        self.installed_forte_dir = utils.get_installed_forte_dir()
        self.exclude_from_writing: Set[str] = set()

        # Directories to be examined to find json schema or installed ontology
        # packages which the user wants to import.
        self.import_dirs: List[str] = []

        # User defined directories are top choices.
        if import_dirs is not None:
            self.import_dirs.extend(import_dirs)

        # The current directory is secondary.
        self.import_dirs.append(os.getcwd())

        spec_base = "forte/ontology_specs"
        forte_spec_dir = get_resource(spec_base, False)

        # Lastly, the Forte installed directory.
        self.import_dirs.append(forte_spec_dir)
        self.exclude_from_writing = set()

        if not generate_all:
            logging.info(
                "Checking existing specification " "directory: %s",
                forte_spec_dir,
            )
            for existing_spec in os.listdir(forte_spec_dir):
                if existing_spec.endswith(".json"):
                    logging.info(
                        "Forte library contains %s, " "will skip this one.",
                        existing_spec,
                    )
                    self.exclude_from_writing.add(
                        os.path.join(spec_base, existing_spec)
                    )

    @no_type_check
    def initialize_top_entries(
        self, manager: ImportManager, base_ontology_module: ModuleType
    ):
        """
        Parses the file corresponding to `base_ontology_module` -
        (1) Imports the imports defined by the base file,
        (2) Imports the public API defined by the base file in it's `__all__`
        attribute. The imports are added to the import manager.

        (3) Extracts the name and inheritance of the class definitions and
        populates `self.top_to_core_entries`,
        (4) Extracts `__init__` arguments of class definitions and populates
        `self.top_init_args`
        (5) Includes type annotations for the `__init__` arguments.

        Args:
            manager: The import manager to be populated.
            base_ontology_module: File path of the module to be parsed.

        Returns:
        """
        tree = None
        with open(
            base_ontology_module.__file__, "r", encoding="utf-8"
        ) as base_ontology_file:
            tree = ast.parse(base_ontology_file.read())

        if tree is None:
            raise RuntimeError(
                f"Fail to load AST Tree from {base_ontology_module.__file__}"
            )

        base_module_name = base_ontology_module.__name__

        # Record a map from the import name to the full name.
        full_names = {}

        for elem in tree.body:
            # Adding all the imports.
            if isinstance(elem, ast.Import):
                for import_ in elem.names:
                    as_name = import_.asname
                    import_name = import_.name if as_name is None else as_name
                    full_names[import_name] = import_.name
                    manager.add_object_to_import(import_.name)

            if isinstance(elem, ast.ImportFrom):
                for import_ in elem.names:
                    full_names[import_.name] = f"{elem.module}.{import_.name}"
                    full_name = f"{elem.module}.{import_.name}"
                    manager.add_object_to_import(full_name)

            # Adding all the module objects defined in `__all__` to imports.
            if isinstance(elem, ast.Assign) and len(elem.targets) > 0:
                if elem.targets[0].id == "__all__":
                    full_names.update(
                        [
                            (
                                name.s,
                                f"{base_ontology_module.__name__}.{name.s}",
                            )
                            for name in elem.value.elts
                        ]
                    )

                    for name in elem.value.elts:
                        full_class_name = f"{base_module_name}.{name.s}"
                        manager.add_object_to_import(full_class_name)

            # Adding `__init__` arguments for each class
            if isinstance(elem, ast.ClassDef):
                # Adding base names for each class
                elem_base_names = set()
                if elem.bases is not None and len(elem.bases) > 0:
                    for elem_base in elem.bases:
                        while isinstance(elem_base, ast.Subscript):
                            # TODO: Doesn't handle typed class well.
                            elem_base_names.add(elem_base.slice.value.id)
                            elem_base_names.add(elem_base.value.id)
                            elem_base = elem_base.slice.value
                        elem_base_names.add(elem_base.id)
                init_func = None

                for func in elem.body:
                    if (
                        isinstance(func, ast.FunctionDef)
                        and func.name == "__init__"
                    ):
                        init_func = func
                        break

                if init_func is None:
                    warnings.warn(
                        f"No `__init__` function found in the class"
                        f" {elem.name} of the module "
                        f"{base_ontology_module}."
                    )
                else:
                    full_ele_name = full_names[elem.name]

                    # Assuming no variable args and keyword only args present in
                    # the base ontology module
                    for arg in init_func.args.args:
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
                                arg_ann.id = full_names[module]
                                manager.add_object_to_import(arg_ann.id)

                                # Convert from PackType to more concrete pack
                                # type, such as DataPack or MultiPack.
                                if arg_ann.id == PACK_TYPE_CLASS_NAME:
                                    pack_class = hardcoded_pack_map(
                                        full_ele_name
                                    )
                                    manager.add_object_to_import(pack_class)
                                    arg_ann.id = pack_class

                    self.top_to_core_entries[full_ele_name] = elem_base_names
                    self.base_entry_lookup[full_ele_name] = full_ele_name
                    self.top_init_args[full_ele_name] = init_func.args

    def generate(
        self,
        spec_path: str,
        destination_dir: str = os.getcwd(),
        is_dry_run: bool = False,
        merged_path: Optional[str] = None,
        lenient_prefix=False,
        namespace_depth: int = -1,
    ) -> Optional[str]:
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
            merged_path: if a path is provided, a merged ontology file will
                be written at this path.
            lenient_prefix: if `True`, will not enforce the entry name to
                match a known prefix.
            namespace_depth: set an integer argument namespace_depth to allow
              customized number of levels of namespace packaging.
              The generation of __init__.py for all the directory
              levels above namespace_depth will be disabled.
              For example, if we have an ontology level1.levle2.level3.
              something and namespace_depth=2, then we remove __init__.py
              under level1 and level1/level2 while keeping __init__.py under
              level1/level2/level3.
              When namespace_depth<=0, we just disable namespace packaging
              and include __init__.py in all directory levels.

        Returns:
            Directory path in which the modules are created: either one of
            the temporary directory or `destination_dir`.

        """
        # Update the list of directories to be examined for imported configs
        self.import_dirs.append(os.path.dirname(os.path.realpath(spec_path)))

        merged_schemas: List[Dict] = []
        merged_prefixes: List[str] = []

        # Generate ontology classes for the input json config and the configs
        # it is dependent upon.
        try:
            self.parse_ontology_spec(
                spec_path,
                merged_schema=merged_schemas,
                merged_prefixes=merged_prefixes,
                lenient_prefix=lenient_prefix,
            )
        except OntologySpecError:
            logging.error("Error at parsing [%s]", spec_path)
            raise

        # Now generate all data.

        # A temporary directory to save the generated file structure until the
        # generation is completed and verified.
        tempdir = tempfile.mkdtemp()
        # Starting from here, we won't add any more modules to import.
        self.import_managers.fix_all_modules()

        logging.info("Working on %s", spec_path)
        for writer in self.module_writers.writers():
            logging.info("Writing module: %s", writer.module_name)
            writer.write(tempdir, destination_dir, namespace_depth)
            logging.info("Done writing.")

        if merged_path is not None:
            logging.info("Writing merged schema at %s", merged_path)
            merged_config = {
                "name": "all_ontology",
                "definitions": merged_schemas,
                "additional_prefixes": list(set(merged_prefixes)),
            }
            with open(merged_path, "w", encoding="utf-8") as out:
                json.dump(merged_config, out, indent=2)
            logging.info("Done writing.")

        # When everything is successfully completed, copy the contents of
        # `self.tempdir` to the provided folder.
        if not is_dry_run:
            generated_top_dirs = set(utils.get_top_level_dirs(tempdir))
            for existing_top_dir in utils.get_top_level_dirs(destination_dir):
                if existing_top_dir in generated_top_dirs:
                    logging.warning(
                        "The directory with the name "
                        "%s is already present in "
                        "%s. New files will be merge into the "
                        "existing directory. Note that in this "
                        "case, the namespace depth may not take "
                        "effect.",
                        existing_top_dir,
                        destination_dir,
                    )

            utils.copytree(
                tempdir,
                destination_dir,
                ignore_pattern_if_file_exists="*/__init__.py",
            )
            return destination_dir

        return tempdir

    def visit_ontology_imports(
        self,
        import_path: str,
        visited_paths: Optional[Dict[str, bool]] = None,
        rec_visited_paths: Optional[Dict[str, bool]] = None,
    ) -> Optional[Tuple[str, Dict[str, bool], Dict[str, bool]]]:
        # Initialize the visited dicts when the function is called for the
        # first time.
        if visited_paths is None:
            visited_paths = defaultdict(lambda: False)

        if rec_visited_paths is None:
            rec_visited_paths = defaultdict(lambda: False)

        # Check for import cycles
        if rec_visited_paths[import_path]:
            raise OntologyAlreadyGeneratedException(
                f"Ontology corresponding to {import_path} already "
                f"generated, cycles not permitted, aborting"
            )

        # If the ontology is already generated, need not generate it again
        if visited_paths[import_path]:
            return None

        # Add the json_file path to the visited dictionaries
        visited_paths[import_path] = True
        rec_visited_paths[import_path] = True

        # Validate and load the ontology specification.
        try:
            utils.validate_json_schema(import_path)
        except Exception as exception:
            if type(exception).__name__.split(".", maxsplit=1)[
                0
            ] == jsonschema.__name__ and hasattr(exception, "message"):
                raise OntologySpecValidationError() from exception
            raise

        return import_path, visited_paths, rec_visited_paths

    def find_import_path(self, import_path):
        for import_dir in self.import_dirs:
            full_spec_path = os.path.join(import_dir, import_path)
            if os.path.exists(full_spec_path):
                return full_spec_path

        raise OntologySourceNotFoundException(
            "Cannot find import [%s]." % import_path
        )

    def parse_ontology_spec(
        self,
        ontology_path: str,
        merged_schema: List[Dict],
        merged_prefixes: List[str],
        visited_paths: Optional[Dict[str, bool]] = None,
        rec_visited_paths: Optional[Dict[str, bool]] = None,
        lenient_prefix=False,
    ):
        r"""Performs a topological traversal on the directed graph formed by the
        imported json configs. While processing each config, it first generates
        the classes corresponding to the entries of the imported configs, then
        imports the generated python classes to generate the classes
        corresponding to the entries of `json_file_path`.
        Args:
            ontology_path: Path to the ontology.
            merged_schema: A list that store all the schema definitions.
            merged_prefixes: A list of prefixes from all schemas.
            visited_paths: Keeps track of the json configs already processed.
            rec_visited_paths: Keeps track of the current recursion stack, to
                detect, and throw error if any cycles are present.
            lenient_prefix: Whether to relax the requirement on the prefix.
        Returns:
        """
        import_info = self.visit_ontology_imports(
            ontology_path, visited_paths, rec_visited_paths
        )

        if import_info is None:
            return

        json_file_path, visited_paths, rec_visited_paths = import_info

        with open(json_file_path, "r", encoding="utf-8") as f:
            spec_dict = json.load(f)

        # Parse imported ontologies. Users can import them via a path relative
        # to the PYTHONPATH.
        relative_imports: Set[str] = set(
            spec_dict.get(SchemaKeywords.imports, [])
        )

        for rel_import in relative_imports:
            full_pkg_path: str = self.find_import_path(rel_import)
            logging.info("Imported ontology at: %s", full_pkg_path)
            self.parse_ontology_spec(
                full_pkg_path,
                merged_schema,
                merged_prefixes,
                visited_paths=visited_paths,
                rec_visited_paths=rec_visited_paths,
                lenient_prefix=lenient_prefix,
            )

        # Once the ontology for all the imported files is generated, generate
        # ontology of the current file.
        # Print relative json path in the ontology if the current directory is
        # the installation directory - example, when running the test cases
        curr_forte_dir = utils.get_current_forte_dir()

        print_json_file = json_file_path
        if self.installed_forte_dir is not None and os.path.samefile(
            curr_forte_dir, self.installed_forte_dir
        ):
            print_json_file = os.path.relpath(json_file_path, curr_forte_dir)

        self.parse_schema(
            spec_dict,
            print_json_file,
            merged_schema,
            merged_prefixes,
            lenient_prefix,
        )

        rec_visited_paths[json_file_path] = False

    def parse_schema_for_no_import_onto_specs_file(
        self,
        ontology_path: str,
        ontology_dict: Dict,
        lenient_prefix=False,
        merged_entry_tree: Optional[EntryTree] = None,
    ):
        r"""Function to populate the `merged_entry_tree` after reading
            ontology from the input json file.

        Args:
            ontology_path: The path to the input ontology specification file,
                which should be a json file, and it should have all the entries
                inside with no import as key.
            ontology_dict: The loaded dictionary of ontology specifications
            lenient_prefix: Whether to remove the constraint on the prefix set.
            merged_entry_tree: an EntryTree type object and if it's not`None`
                then after running this function, all the entries from
                ontology specification file would be parsed into a tree
                structure with parent and children entries to represent
                the relationship.

        Returns:

        """
        # Assume no import key in the ontology file
        if SchemaKeywords.imports in ontology_dict.keys():
            raise OntologySpecValidationError(
                "The system cannot build entry trees from an ontology file "
                "that imports other ontologies. But we find `import` keywords "
                f"in the provided ontology {ontology_path}"
                " Please provide a merged ontology file with all imports "
                "resolved by using the command "
                "`generate_ontology create --merged_path`"
            )

        merged_schema: List[Dict] = []
        merged_prefixes: List[str] = []

        self.parse_schema(
            ontology_dict,
            ontology_path,
            merged_schema,
            merged_prefixes,
            lenient_prefix,
            merged_entry_tree,
        )

    def parse_schema(
        self,
        schema: Dict,
        source_json_file: str,
        merged_schema: List[Dict],
        merged_prefixes: List[str],
        lenient_prefix=False,
        merged_entry_tree: Optional[EntryTree] = None,
    ):
        r"""Generates ontology code for a parsed schema extracted from a
        json config. Appends entry code to the corresponding module. Creates a
        new module file if module is generated for the first time.

        Args:
            schema: Ontology dictionary extracted from a json config.
            source_json_file: Path of the source json file.
            merged_schema: The merged schema is used to remember all
                definitions during parsing.
            merged_prefixes: To remember all prefixes encountered during
                parsing.
            lenient_prefix: Whether to remove the constraint on the prefix set.
            merged_entry_tree: an EntryTree type object and if it's not`None`
                then after running this function, all the entries from
                ontology specification file would be parsed into a tree
                structure with parent and children entries to represent
                the relationship.
        Returns:
            Modules to be imported by dependencies of the current ontology.
        """
        entry_definitions: List[Dict] = schema[SchemaKeywords.definitions]
        merged_schema.extend(entry_definitions)

        if SchemaKeywords.prefixes in schema:
            merged_prefixes.extend(schema[SchemaKeywords.prefixes])

        allowed_packages = set(
            schema.get(SchemaKeywords.prefixes, []) + [DEFAULT_PREFIX]
        )
        sorted_prefixes = analyze_packages(allowed_packages)

        file_desc = file_header(
            schema.get(SchemaKeywords.description, ""),
            schema.get(SchemaKeywords.ontology_name, ""),
        )

        for definition in entry_definitions:
            raw_entry_name = definition[SchemaKeywords.entry_name]
            validate_entry(raw_entry_name, sorted_prefixes, lenient_prefix)

            if raw_entry_name in self.allowed_types_tree:
                warnings.warn(
                    f"Class {raw_entry_name} already present in the "
                    f"ontology, will be overridden.",
                    DuplicateEntriesWarning,
                )
            self.allowed_types_tree[raw_entry_name] = set()

            # Add the entry definition to the import managers.
            # This time adding to the root manager so everyone can access it
            # if needed, but they will only appear in the import list when
            # requested.
            # Entry class should be added to the imports before the attributes
            # to be able to used as the attribute type for the same entry.
            self.import_managers.root.add_object_to_import(raw_entry_name)

            # Get various parts of the entry name.
            en = EntryName(raw_entry_name)
            # First add the entry, and then parse the attribute. In this
            #  order, we can avoid some incorrect warning.
            self.import_managers.get(en.module_name).add_defining_objects(
                raw_entry_name
            )
            entry_item, properties = self.parse_entry(en, definition)

            # Get or set module writer only if the ontology to be generated
            # is not already installed.
            if source_json_file not in self.exclude_from_writing:
                module_writer = self.module_writers.get(en.module_name)
                module_writer.set_description(file_desc)
                module_writer.source_file = source_json_file
                # Add entry item to the writer.
                module_writer.add_entry(en, entry_item)

            # Adding entry attributes to the allowed types for validation.
            for property in properties:
                property_name = property[0]
                # Check if the name is allowed.
                if not property_name.isidentifier():
                    raise InvalidIdentifierException(
                        f"The property name: {property_name} is not a valid "
                        f"python identifier."
                    )

                if property_name in set(
                    val[0] for val in self.allowed_types_tree[en.class_name]
                ):
                    warnings.warn(
                        f"Attribute type for the entry {en.class_name} "
                        f"and the attribute {property_name} already present in "
                        f"the ontology, will be overridden",
                        DuplicatedAttributesWarning,
                    )
                self.allowed_types_tree[en.class_name].add(property)
            # populate the entry tree based on information
            if merged_entry_tree is not None:
                curr_entry_name = en.class_name
                parent_entry_name = definition["parent_entry"]
                curr_entry_attributes = self.allowed_types_tree[en.class_name]
                merged_entry_tree.add_node(
                    curr_entry_name, parent_entry_name, curr_entry_attributes
                )

    def cleanup_generated_ontology(
        self, path, is_forced=False
    ) -> (Tuple[bool, Optional[str]]):
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

        rel_paths = dir_util.copy_tree(path, "", dry_run=1)
        rel_paths = [
            os.path.dirname(file)
            for file in rel_paths
            if os.path.basename(file).startswith(AUTO_GEN_FILENAME)
        ]

        del_dir = None
        if not is_forced:
            curr_time_str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
            del_dir = os.path.join(
                os.path.dirname(path), AUTO_DEL_FILENAME, curr_time_str
            )
            for rel_path in rel_paths:
                joined_path = os.path.join(del_dir, rel_path)
                Path(joined_path).mkdir(parents=True, exist_ok=True)
        rel_paths += [""]
        return (
            self._cleanup_generated_ontology(path, "", del_dir, rel_paths),
            del_dir,
        )

    def _cleanup_generated_ontology(
        self, outer_path, relative_path, delete_dir, allowed_relative_paths
    ) -> bool:
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
        dst_dir = (
            os.path.join(delete_dir, os.path.dirname(relative_path))
            if delete_dir is not None
            else None
        )

        if os.path.isfile(path):
            # path is a file type
            # delete .generated marker files and automatically generated files
            is_empty = os.path.basename(path).startswith(AUTO_GEN_FILENAME)
            if not is_empty and os.access(path, os.R_OK):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        if lines[0].startswith(f"# {AUTO_GEN_SIGNATURE}"):
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
                if not self._cleanup_generated_ontology(
                    outer_path,
                    child_rel_path,
                    delete_dir,
                    allowed_relative_paths,
                ):
                    is_empty = False
            if is_empty:
                if delete_dir is not None:
                    dir_util.copy_tree(path, dst_dir)
                os.rmdir(path)
        return is_empty

    def replace_annotation(self, entry_name: EntryName, func_args):
        this_manager = self.import_managers.get(entry_name.module_name)
        for arg in func_args.args:
            if arg.annotation is not None:
                arg_ann = arg.annotation
                # Handling the type name for cases like Optional[X]
                while isinstance(arg_ann, ast.Subscript):
                    # The types for arg_ann and so on are in typed_ast._ast3,
                    # these types are protected hence hard to be used here.

                    short_ann_name = arg_ann.value.id  # type: ignore
                    full_ann_name: str = this_manager.get_name_to_use(
                        short_ann_name
                    )

                    arg_ann.value.id = full_ann_name  # type: ignore
                    arg_ann = arg_ann.slice.value  # type: ignore

                # Handling the type name for arguments.
                arg_ann.id = this_manager.get_name_to_use(arg_ann.id)

    def construct_init(self, entry_name: EntryName, base_entry: str):
        base_init_args = self.top_init_args[base_entry]
        custom_init_args = copy.deepcopy(base_init_args)

        self.replace_annotation(entry_name, custom_init_args)
        custom_init_args_str = as_init_str(custom_init_args)
        return custom_init_args_str

    def parse_entry(
        self, entry_name: EntryName, schema: Dict
    ) -> Tuple[EntryDefinition, List[Tuple[str, str]]]:
        """
        Args:
            entry_name: Object holds various name form of the entry.
            schema: Dictionary containing specifications for an entry.

        Returns: extracted entry information: entry package string, entry
        filename, entry class entry_name, generated entry code and a list
        of tuples where each element in the list represents the an attribute
        in the entry and its corresponding type.
        """
        this_manager = self.import_managers.get(entry_name.module_name)

        # Determine the parent entry of this entry.
        parent_entry: str = schema[SchemaKeywords.parent_entry]

        if parent_entry.startswith(TOP_MOST_MODULE_NAME):
            raise ParentEntryNotSupportedException(
                f"The parent entry {parent_entry} cannot be directly inherited,"
                f" please inherit a type from {top.__name__} or your own"
                f" ontology."
            )

        if not this_manager.is_imported(parent_entry):
            raise ParentEntryNotDeclaredException(
                f"The parent entry {parent_entry} is not declared. It is "
                f"neither in the base entries nor in custom entries. "
                f"Please check them ontology specification, and make sure the "
                f"entry is defined before this."
            )

        base_entry: Optional[str] = self.find_base_entry(
            entry_name.class_name, parent_entry
        )

        if base_entry is None:
            raise OntologySpecError(
                f"Cannot find the base entry for entry "
                f"{entry_name.class_name} and {parent_entry}"
            )

        if base_entry not in self.top_init_args:
            raise ParentEntryNotSupportedException(
                f"Cannot add {entry_name.class_name} to the ontology as "
                f"it's parent entry {parent_entry} is not supported. This is "
                f"likely that the entries are not inheriting the allowed types."
            )

        # Take the property definitions of this entry.
        properties: List[Dict] = schema.get(SchemaKeywords.attributes, [])

        this_manager = self.import_managers.get(entry_name.module_name)

        # Validate if the parent entry is present.
        if not this_manager.is_known_name(parent_entry):
            raise ParentEntryNotDeclaredException(
                f"Cannot add {entry_name.class_name} to the ontology as "
                f"it's parent entry {parent_entry} is not present "
                f"in the ontology."
            )

        parent_entry_use_name = this_manager.get_name_to_use(parent_entry)

        property_items, property_names = [], []
        for prop_schema in properties:
            # TODO: add test

            # the prop attributes will store the properties of each attribute
            # of the the entry defined by the ontology. The properties are
            # the name of the attribute and its data type.
            prop = (prop_schema["name"], prop_schema["type"])

            if prop_schema["name"] in RESERVED_ATTRIBUTE_NAMES:
                raise InvalidIdentifierException(
                    f"The attribute name {prop_schema['name']} is reserved and cannot be "
                    f"used, please consider changed the name. The list of "
                    f"reserved name strings are "
                    f"{RESERVED_ATTRIBUTE_NAMES}"
                )

            property_names.append(prop)
            property_items.append(self.parse_property(entry_name, prop_schema))

        # For special classes that requires a constraint.
        core_bases: Set[str] = self.top_to_core_entries[base_entry]
        entry_constraint_keys: Dict[str, str] = {}
        if any(item == "BaseLink" for item in core_bases):
            entry_constraint_keys = DEFAULT_CONSTRAINTS_KEYS["BaseLink"]
        elif any(item == "BaseGroup" for item in core_bases):
            entry_constraint_keys = DEFAULT_CONSTRAINTS_KEYS["BaseGroup"]

        class_att_items: List[ClassTypeDefinition] = []
        for schema_key, class_key in entry_constraint_keys.items():
            if schema_key in schema:
                constraint_type_ = schema[schema_key]
                constraint_type_name = this_manager.get_name_to_use(
                    constraint_type_
                )

                if constraint_type_name is None:
                    raise TypeNotDeclaredException(
                        f"The type {constraint_type_} is not defined but it is "
                        f"specified in {schema_key} of the definition of "
                        f"{schema['entry_name']}. Please define them before "
                        f"this entry type."
                    )

                # TODO: cannot handle constraints that contain self-references.
                # self_ref = entry_name.class_name == constraint_type_

                class_att_items.append(
                    ClassTypeDefinition(class_key, constraint_type_name)
                )

        # TODO: Can assign better object type to Link and Group objects
        custom_init_arg_str: str = self.construct_init(entry_name, base_entry)

        entry_item = EntryDefinition(
            name=entry_name.name,
            class_type=parent_entry_use_name,
            init_args=custom_init_arg_str,
            properties=property_items,
            class_attributes=class_att_items,
            description=schema.get(SchemaKeywords.description, None),
        )

        return entry_item, property_names

    def parse_ndarray(
        self,
        manager: ImportManager,
        schema: Dict,
        att_name: str,
        desc: str,
    ):
        ndarray_dtype = None
        if SchemaKeywords.ndarray_dtype in schema:
            ndarray_dtype = schema[SchemaKeywords.ndarray_dtype]

        ndarray_shape = None
        if SchemaKeywords.ndarray_shape in schema:
            ndarray_shape = schema[SchemaKeywords.ndarray_shape]

        if ndarray_dtype is None or ndarray_shape is None:
            warnings.warn(
                "Either dtype or shape is not specified."
                " It is recommended to specify both of them."
            )

        default_val = None
        if ndarray_dtype and ndarray_shape:
            default_val = ndarray(ndarray_shape, dtype=ndarray_dtype)

        return NdArrayProperty(
            manager,
            att_name,
            ndarray_dtype,
            ndarray_shape,
            description=desc,
            default_val=default_val,
        )

    def parse_dict(
        self,
        manager: ImportManager,
        schema: Dict,
        entry_name: EntryName,
        att_name: str,
        att_type: str,
        desc: str,
    ):
        if (
            SchemaKeywords.dict_key_type not in schema
            or SchemaKeywords.dict_value_type not in schema
        ):
            raise TypeNotDeclaredException(
                f"Item type of the attribute {att_name} for the entry "
                f" {entry_name.class_name} not declared. This attribute is "
                f"a composite type: {att_type}, it should have a "
                f"{SchemaKeywords.dict_key_type} and "
                f"{SchemaKeywords.dict_value_type}."
            )

        key_type = schema[SchemaKeywords.dict_key_type]
        if not valid_composite_key(key_type):
            raise UnsupportedTypeException(
                f"Key type {key_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is not supported, we only support a "
                f"limited set of keys."
            )

        value_type = schema[SchemaKeywords.dict_value_type]
        if is_composite_type(value_type):
            # Case of nested.
            raise UnsupportedTypeException(
                f"Item type {value_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is a composite type, we do not support "
                f"nested composite type."
            )

        if not manager.is_known_name(value_type):
            # Case of unknown.
            raise TypeNotDeclaredException(
                f"Item type {value_type} for the entry "
                f"{entry_name.name} of the attribute {att_name} "
                f"not declared in ontology."
            )

        # Make sure the import of these related types are handled.
        manager.add_object_to_import(value_type)

        self_ref = entry_name.class_name == value_type

        default_val: Dict = {}

        return DictProperty(
            manager,
            att_name,
            key_type,
            value_type,
            description=desc,
            default_val=default_val,
            self_ref=self_ref,
        )

    def parse_list(
        self,
        manager: ImportManager,
        schema: Dict,
        entry_name: EntryName,
        att_name: str,
        att_type: str,
        desc: str,
    ) -> ListProperty:
        if SchemaKeywords.element_type not in schema:
            raise TypeNotDeclaredException(
                f"Item type for the attribute {att_name} of the entry "
                f"[{entry_name.class_name}] not declared. This attribute is "
                f"a composite type: {att_type}, it should have a "
                f"{SchemaKeywords.element_type}."
            )

        item_type = schema[SchemaKeywords.element_type]
        if is_composite_type(item_type):
            # Case of nested.
            raise UnsupportedTypeException(
                f"Item type {item_type} for entry {entry_name.name}'s "
                f"attribute {att_name} is a composite type, we do not support "
                f"nested composite type."
            )

        if not manager.is_known_name(item_type):
            # Case of unknown.
            raise TypeNotDeclaredException(
                f"Item type {item_type} for the entry "
                f"{entry_name.name} of the attribute {att_name} "
                f"not declared in ontology."
            )

        # Make sure the import of these related types are handled.
        manager.add_object_to_import(item_type)

        self_ref = entry_name.class_name == item_type

        return ListProperty(
            manager,
            att_name,
            item_type,
            description=desc,
            default_val=[],
            self_ref=self_ref,
        )

    def parse_non_composite(
        self,
        manager: ImportManager,
        att_name: str,
        att_type: str,
        desc: str,
        default_val: str,
        self_ref: bool = False,
    ) -> NonCompositeProperty:
        manager.add_object_to_import("typing.Optional")

        return NonCompositeProperty(
            manager,
            att_name,
            att_type,
            description=desc,
            default_val=default_val,
            self_ref=self_ref,
        )

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
            entry_name.module_name
        )

        # schema type should be present in the validation tree
        # TODO: Remove this hack
        if not manager.is_known_name(att_type):
            raise TypeNotDeclaredException(
                f"Attribute type '{att_type}' for the entry "
                f"'{entry_name.name}' of the attribute '{att_name}' not "
                f"declared in the ontology"
            )

        desc = schema.get(SchemaKeywords.description, None)
        default_val = schema.get(SchemaKeywords.default_value, None)

        # element type should be present in the validation tree
        if att_type in COMPOSITES:
            if att_type == "List":
                return self.parse_list(
                    manager, schema, entry_name, att_name, att_type, desc
                )
            elif att_type == "Dict":
                return self.parse_dict(
                    manager, schema, entry_name, att_name, att_type, desc
                )
            elif att_type == "NdArray":
                return self.parse_ndarray(manager, schema, att_name, desc)
        elif att_type in NON_COMPOSITES or manager.is_imported(att_type):
            self_ref = entry_name.class_name == att_type
            return self.parse_non_composite(
                manager,
                att_name,
                att_type,
                desc,
                default_val,
                self_ref=self_ref,
            )

        raise UnsupportedTypeException(f"{att_type} is not a supported type.")

    def find_base_entry(
        self, this_entry: str, parent_entry: str
    ) -> Optional[str]:
        """Find the `base_entry`. As a side effect, it will populate the
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
        base_entry: Optional[str]
        if parent_entry in self.top_init_args:
            # The top init args contains the objects in the top.py.
            base_entry = parent_entry
        else:
            base_entry = self.base_entry_lookup.get(parent_entry, None)

        if base_entry is not None:
            self.base_entry_lookup[this_entry] = base_entry

        return base_entry
