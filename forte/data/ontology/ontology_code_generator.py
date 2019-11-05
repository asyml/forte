"""
    Module to automatically generate python ontology given json file
    Performs a preliminary check of dependencies
"""
import os
import json
import logging
import tempfile
from pathlib import Path
from collections import defaultdict
from distutils import dir_util

from types import ModuleType
from typing import Dict, List, Optional, Tuple, Set

import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse

from forte.data.ontology import utils, top
from forte.data.ontology.code_generation_util import (
    BasicItem, CompositeItem, DefinitionItem, FileItem)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class OntologyCodeGenerator:
    """
    Class to generate python ontology given ontology config in json format
    Salient Features -
        (1) Generates a class for each entry in the module corresponding to
        the defined entry package
        (2) Dependencies to other json configs through the json `imports`
    Example:
        >>> destination_dir = OntologyCodeGenerator().generate_ontology(
        'test/example_ontology_config.json')
    """
    AUTO_GEN_CONST = '***automatically_generated***'

    def __init__(self,
                 json_dir_paths: Optional[List[str]] = None,
                 top_level_dir: Optional[str] = None,
                 ontology_base_module: ModuleType = top):
        """
        Args:
            ontology_base_module: The entries of the `base_ontology_module`
            serve as ancestors of the user-defined entries.
            json_dir_paths: Additional user provided paths to search the
            imported json configs from. By default paths provided in the json
            configs and the current working directory would be searched.
            indent: Number of indent required in the indentation.
        """
        self.base_ontology_module = ontology_base_module
        self.top_level_dir = "generated-files" \
            if top_level_dir is None else top_level_dir

        # Builtin and local imports required in the generated python modules.
        self.required_imports: List[str] = [
            'typing',
            'forte.data.data_pack',
            ontology_base_module.__name__]

        # Special comments to be added to disable checking.
        self.ignore_errors: List[str] = [
            f'# {self.AUTO_GEN_CONST}',
            '# flake8: noqa',
            '# mypy: ignore-errors',
            '# pylint: skip-file']

        # Mapping from entries parsed from the `base_ontology_module`
        # (default is `top.py`), to their `__init__` arguments.
        self.top_init_args: Dict[str, str] = {}
        self.initialize_top_entries()

        # Mapping from user-defined entries to their ancestor entry present in
        # `self.top_init_args`.
        self.user_to_base_entry: Dict[str, str] = {}

        # Mapping from reference string to corresponding namespace
        self.ref_to_namespace: Dict[str, str] = {}

        # Adjacency list to store the allowed types (in-built or user-defined),
        # and their attributes (if any) in order to validate the attribute
        # types.
        self.allowed_types_tree: Dict[str, Set] = {}
        for type_str in {*BasicItem.TYPES, *CompositeItem.TYPES}:
            self.allowed_types_tree[type_str] = set()
            self.ref_to_namespace[type_str] = type_str

        # A temporary directory to save the generated file structure until the
        # generation is completed and verified.
        self.tempdir: str

        # Directories to be examined to find json files for user-defined config
        # imports.
        self.json_paths: List[str] = [] \
            if json_dir_paths is None else json_dir_paths

    def initialize_top_entries(self):
        """
        Parses the base file corresponding to `self.base_ontology_module` -
        (1) Imports the imports defined by the base file,
        (2) Imports the public API defined by by the base file in it's `__all__`
        attribute,
        (3) Extracts the class definitions and their `__init__` arguments,
        (4) Includes type annotations for the `__init__` arguments.

        Initialises `self.top_init_args` with mapping from an entry class name
        to `__init__` arguments.
        """
        tree = ast.parse(open(self.base_ontology_module.__file__).read())

        imports = {}

        for elem in tree.body:
            # Adding all the imports.
            if isinstance(elem, ast.Import):
                for import_ in elem.names:
                    import_name = import_.name.split('.')[-1]
                    imports[import_name] = import_.name

            if isinstance(elem, ast.ImportFrom):
                for import_ in elem.names:
                    imports[import_.name] = f"{elem.module}.{import_.name}"

            # Adding all the module objects defined in `__all__` to imports.
            if isinstance(elem, ast.Assign) and len(elem.targets) > 0:
                if elem.targets[0].id == '__all__':
                    imports.update(
                        [(name.s,
                          f"{self.base_ontology_module.__name__}.{name.s}")
                         for name in elem.value.elts])

            # Adding `__init__` arguments for each class
            if isinstance(elem, ast.ClassDef):
                init_func = None
                for func in elem.body:
                    if isinstance(func, ast.FunctionDef) and \
                            func.name == '__init__':
                        init_func = func
                        break
                if init_func is None:
                    raise Warning(f"No `__init__` function found in the class "
                                  f"{elem.name} of the module "
                                  f"{self.base_ontology_module}.")
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
                    args_str = args[1].strip()\
                        .replace('\n', '').replace('  ', '')

                    self.top_init_args[imports[elem.name]] = args_str

    def generate_ontology(self, base_json_file_path: str,
                          destination_dir: Optional[str] = None,
                          is_dry_run: bool = False) -> Optional[str]:
        """
            Function to generate and save the python ontology code after reading
            ontology from the input json file.
            Args:
                base_json_file_path: The base json config file.
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
                self.ref_to_namespace[obj_str] = full_obj_str

        # Generate ontology classes for the input json config and the configs
        # it is dependent upon.
        self.parse_ontology(base_json_file_path)

        # When everything is successfully completed, copy the contents of
        # `self.tempdir` to the provided folder.
        if not is_dry_run:
            destination_dir = os.getcwd() if destination_dir is None \
                else destination_dir
            dest_path = os.path.join(destination_dir, self.top_level_dir)
            generated_top_dirs = set(utils.get_top_level_dirs(self.tempdir))
            for existing_top_dir in utils.get_top_level_dirs(dest_path):
                if existing_top_dir in generated_top_dirs:
                    raise ValueError(f"The directory with the name "
                                     f"{existing_top_dir} is already present in"
                                     f"{dest_path}.")

            dir_util.copy_tree(self.tempdir, dest_path)

            return dest_path

        return self.tempdir

    def parse_ontology(self, json_file_path: str,
                       visited_paths: Optional[Dict[str, bool]] = None,
                       rec_visited_paths: Optional[Dict[str, bool]] = None)\
            -> List[str]:
        """
        Performs a topological traversal on the directed graph formed by the
        imported json configs. While processing each config, it first generates
        the classes corresponding to the entries of the imported configs, then
        imports the generated python classes to generate the classes
        corresponding to the entries of `json_file_path`.
        Args:
            json_file_path: The current json config to be processed.
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
            curr_str = f.read()
        curr_dict = json.loads(curr_str)

        # Extract imported json files and generate ontology for them.
        json_imports_dict: Dict = curr_dict.get("imports", {})

        modules_to_import: List[str] = []

        for import_ref in json_imports_dict:
            import_schema = json_imports_dict[import_ref]
            import_namespace = import_schema["type"]
            self.ref_to_namespace[import_ref] = import_namespace
            import_file = f"{import_namespace.split('.')[-1]}_config.json"
            import_json_file = utils.search_in_dirs(import_file,
                                                    self.json_paths)
            if import_json_file is None:
                raise ValueError(f"Ontology corresponding to {import_file} not"
                                 f"found in the current directory or the "
                                 f"directory of original json config.")
            if import_json_file in rec_visited_paths:
                raise ValueError(f"Ontology corresponding to {import_json_file}"
                                 f" already generated, cycles not permitted, "
                                 f"aborting")
            elif import_json_file not in visited_paths:
                modules_to_import.extend(self.parse_ontology(
                    import_json_file, visited_paths, rec_visited_paths))

        # once the ontologies for all the imported files is generated, generate
        # ontology of the current file
        modules_to_import = self.parse_config(curr_dict, modules_to_import)
        rec_visited_paths[json_file_path] = False
        return modules_to_import

    def parse_config(self, schema: Dict, modules_to_import: List[str]) \
            -> List[str]:
        """
        Generates ontology code for ontology dictionary extracted from a json
        config. Appends entry code to the corresponding module. Creates a new
        module file if module is generated for the first time.
        Args:
            schema: Ontology dictionary extracted from a json config.
            modules_to_import: Dependencies to be imported by generated modules.

        Returns:
            Modules to be imported by dependencies of the current ontology.
        """
        entry_definitions: Dict[str, Dict] = schema["definitions"]

        new_modules_to_import = []
        for ref_name in entry_definitions:
            definition = entry_definitions[ref_name]
            full_name = definition["namespace"]
            name_split = full_name.rsplit('.', 2)
            if len(name_split) == 2:
                name_split = ['ft.onto'] + name_split
            pkg, filename, name = name_split
            self.ref_to_namespace[name] = full_name
            if full_name in self.allowed_types_tree:
                raise Warning(f"Class {full_name} already present in the "
                              f"ontology, will be overridden.")
            self.allowed_types_tree[full_name] = set()
            entry_item, properties = self.parse_entry(ref_name, definition)
            module_name: str = f"{pkg}.{filename}"
            class_name: str = f"{module_name}.{name}"

            try:
                # Creating entry directory and file in the tempdir if required.
                entry_dir: str = os.path.join(self.tempdir, *pkg.split('.'))
                entry_file: str = f'{os.path.join(entry_dir, filename)}.py'
                file_desc: str = 'Automatically generated file. ' \
                                 'Do not change manually.'
                all_imports = self.required_imports + modules_to_import
                file_item = FileItem(entry_item, entry_file, self.ignore_errors,
                                     file_desc, all_imports)

                Path(entry_dir).mkdir(parents=True, exist_ok=True)

                # Creating the file if it does not exist.
                with open(entry_file, 'a+') as f:
                    f.write(file_item.to_code(0))

            except ValueError:
                self.cleanup_generated_ontology(self.tempdir)
                raise

            # Modules to be imported by the dependencies.
            new_modules_to_import.append(module_name)

            # Adding entry attributes to the allowed types for validation.
            for property_name in properties:
                if property_name in self.allowed_types_tree[class_name]:
                    raise Warning(
                        f"Attribute type for the entry {class_name} and "
                        f"the attribute {property_name} already present in "
                        f"the ontology, will be overridden")
                self.allowed_types_tree[class_name].add(property_name)

        return new_modules_to_import

    def cleanup_generated_ontology(self, path, is_forced=False):
        """
        Deletes the generated ontology files.
        """
        path = os.path.abspath(path)
        if os.path.isfile(path):
            is_empty = False
            if os.access(path, os.R_OK):
                with open(path, 'r') as f:
                    line = f.readlines()[0]
                    if line.startswith(f'# {self.AUTO_GEN_CONST}'):
                        is_empty = True
            if is_empty:
                os.unlink(path)
        else:
            is_empty = True
            for child in os.listdir(path):
                child_path = os.path.join(path, child)
                if not self.cleanup_generated_ontology(child_path, is_forced):
                    is_empty = False
            if is_empty:
                to_delete = is_forced
                if not is_forced:
                    to_delete_prompt = f"Delete the directory {path}? <y/n>: "
                    to_delete = input(to_delete_prompt).lower() == "y"
                if to_delete:
                    os.rmdir(path)
            log.info("Deleted %s.", path)
        return is_empty

    def parse_entry(self, ref_name: str, schema: Dict) -> Tuple[DefinitionItem,
                                                                List[str]]:
        """
        Args:
            ref_name:
            schema: Dictionary containing specifications for an entry.

        Returns: extracted entry information: entry package string, entry
        filename, entry class entry_name, generated entry code and entry
        attribute names.
        """
        name = schema["namespace"]
        # reading the entry definition dictionary
        parent_entry: str = self.parse_type(schema["type"])

        properties: Dict[str, Dict] = schema["properties"]

        # validate if the entry parent is present in the tree
        if parent_entry not in self.allowed_types_tree:
            raise ValueError(f"Cannot add {name} to the ontology as "
                             f"it's parent entry {parent_entry} is not present "
                             f"in the ontology.")

        base_entry: str = self.get_and_set_base_entry(name, parent_entry)
        init_args: str = self.top_init_args[base_entry]

        property_items = []
        if properties is not None:
            for prop_name in properties:
                prop_schema = properties[prop_name]
                property_items.append(self.parse_property(name, prop_name,
                                                          prop_schema))

        entry_item = DefinitionItem(name=ref_name,
                                    class_type=parent_entry,
                                    init_args=init_args,
                                    properties=property_items,
                                    description=schema.get("description", None))

        return entry_item, list(properties.keys())

    def parse_type(self, type_):
        if isinstance(type_, str):
            return self.ref_to_namespace[type_]
        refs = type_['$ref'].split('/')[1:]
        return f'"{refs[1]}"' if len(refs) == 2 else self.parse_ref_type(refs)

    def parse_ref_type(self, refs: List[str], prefix=''):
        ref = refs[1]
        ref_namespace = self.ref_to_namespace[ref]
        if refs[0] == "imports":
            return self.parse_ref_type(refs[2:], ref_namespace)
        return ref_namespace if prefix == '' else '.'.join([prefix, ref])

    def parse_property(self, entry_name, name, schema):
        type_str = schema["type"]
        type_ = self.parse_type(type_str)

        # schema type should be present in the validation tree
        if type_ not in self.allowed_types_tree:
            raise ValueError(f"Attribute type '{type_}' for the entry "
                             f"'{entry_name}' and the schema '{name}' not "
                             f"declared in the ontology")

        desc = schema.get("description", None)
        default = schema.get("default", None)

        # TODO: Only supports array for now!
        # element type should be present in the validation tree
        if type_str in CompositeItem.TYPES:
            item_type = schema["items"].get("type", schema["items"])
            item_type = self.parse_type(item_type)
            item_type_norm = item_type.replace('"', '')
            if not (item_type_norm in self.allowed_types_tree
                    or self.ref_to_namespace.get(item_type_norm, '')
                    in self.allowed_types_tree
                    or item_type_norm == entry_name):
                raise ValueError(
                    f"Item type {item_type_norm} for the entry {entry_name} "
                    f"of the property {name} not declared in ontology")

            return CompositeItem(name, type_, [item_type], desc, default)

        return BasicItem(name, type_, desc, default)

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
            `base_entry` for the entry `entry_name`.

        Example:
            If the subclass structure is -
            `Token` inherits `Token` inherits `Annotation`.
            The base_entry for both `Token` and `Token` should be `Annotation`.
        """
        if parent_entry in self.top_init_args:
            base_entry: str = parent_entry
        else:
            base_entry = self.user_to_base_entry[parent_entry]
        self.user_to_base_entry[entry_name] = base_entry
        return base_entry
