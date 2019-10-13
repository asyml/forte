"""
    Module to automatically generate python ontology given json file
    Performs a preliminary check of dependencies
"""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
from distutils.dir_util import copy_tree

from types import ModuleType
from typing import Dict, List, Optional, Tuple, Set

import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse

from forte.data.ontology import utils
from forte.data.ontology.base import top


class OntologyCodeGenerator:
    """
    Class to generate python ontology given ontology config in json format
    Salient Features -
        (1) Generates a class for each entry in the module corresponding to
        the defined entry package
        (2) Dependencies to other json configs through the json `import` key
    Example:
        >>> destination_dir = OntologyCodeGenerator().generate_ontology(
        'test/example_ontology_config.json')
    """

    _IMPORTS: str = "imports"
    _ENTRY_DEFINITIONS: str = "entry_definitions"
    _ENTRY_NAME: str = "entry_name"
    _ENTRY_DESC: str = "entry_description"
    _PARENT_ENTRY: str = "parent_entry"
    _ATTRIBUTES: str = "attributes"
    _ATTRIBUTE_NAME: str = "attribute_name"
    _ATTRIBUTE_DESC: str = "attribute_description"
    _ATTRIBUTE_TYPE: str = "attribute_type"
    _ELEMENT_TYPE: str = "element_type"
    _ATTRIBUTE_DEFAULT_VALUE: str = "attribute_default_value"
    _TOP_LEVEL_DIR: str = "generated-files"
    _INDENT: str = ' ' * 4

    _PRIMARY_TYPES = ["bool", "int", "float", "str"]
    _COMPOSITE_TYPES = ["List"]

    def __init__(self,
                 ontology_base_module: ModuleType = top,
                 json_paths: Optional[List[str]] = None,
                 module_paths: Optional[List[str]] = None):
        """
        Args:
            ontology_base_module: The entries of the `base_ontology_module`
            serve as ancestors of the user-defined entries.
            json_paths: Additional user provided paths to search the imported
            json configs from. By default paths provided in the json configs and
            the current working directory would be searched.
            module_paths: Additional user provided paths where python will looks
            for modules and packages.
        """
        self.base_ontology_module = ontology_base_module

        # Builtin and local imports required in the generated python modules.
        self.required_imports: List[str] = [
            'typing',
            'forte.data.data_pack',
            ontology_base_module.__name__]

        # Special comments to be added to disable checking.
        self.ignore_errors: List[str] = [
            '# flake8: noqa',
            '# mypy: ignore-errors',
            '# pylint: disable=line-too-long,trailing-newlines']

        # Mapping from entries parsed from the `base_ontology_module`
        # (default is `top.py`), to their `__init__` arguments.
        self._top_entries: Dict[str, str] = {}
        self._initialize_top_entries()

        # Mapping from user-defined entries to their ancestor entry present in
        # `self._top_entries`.
        self._base_entry_map_seen: Dict[str, str] = {}

        # Adjacency list to store the allowed types (in-built or user-defined),
        # and their attributes (if any) in order to validate the attribute
        # types.
        self._allowed_types_tree: Dict[str, Set] = {}
        self._initialize_primary_types()

        # A temporary directory to save the generated file structure until the
        # generation is completed and verified.
        self._tempdir: str

        # Directories to be examined to find json files for user-defined config
        # imports.
        self._json_paths: Optional[List[str]] = json_paths \
            if json_paths is not None else []

        # Paths to be added to the PYTHONPATH variable to be able to import
        # modules from.
        self._module_paths: List[str] = module_paths \
            if module_paths is not None else []

    def _initialize_primary_types(self):
        """
            Initialize `self._allowed_types_tree` with allowed python built-in
            data types.
        """
        for allowed_type in self._PRIMARY_TYPES + self._COMPOSITE_TYPES:
            self._allowed_types_tree[allowed_type] = set()

    def _initialize_top_entries(self):
        """
        Parses the base file corresponding to `self.base_ontology_module` -
        (1) Imports the imports defined by the base file,
        (2) Imports the public API defined by by the base file in it's `__all__`
        attribute,
        (3) Extracts the class definitions and their `__init__` arguments,
        (4) Includes type annotations for the `__init__` arguments.

        Initialises `self._top_entries` with mapping from an entry class name
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

                    self._top_entries[imports[elem.name]] = args_str

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
        self._initialize_local_directories(base_json_file_path)

        sys.path.extend(self._module_paths)

        # Adding the imported objects to the allowed types.
        for import_module in self.required_imports:
            for obj_str in utils.get_user_objects_from_module(import_module):
                full_obj_str = f"{import_module}.{obj_str}"
                if full_obj_str in self._allowed_types_tree:
                    raise Warning(f"The object {full_obj_str} of the module "
                                  f"{import_module} is already added in the "
                                  f"ontology, will be overridden.")
                self._allowed_types_tree[full_obj_str] = set()

        # Generate ontology classes for the input json config and the configs
        # it is dependent upon.
        self._generate_ontology(base_json_file_path)

        # When everything is successfully completed, copy the contents of
        # `self._tempdir` to the provided folder.
        if not is_dry_run:
            destination_dir = os.getcwd() if destination_dir is None \
                else destination_dir
            dest_path = os.path.join(destination_dir, self._TOP_LEVEL_DIR)
            generated_top_dirs = set(utils.get_top_level_dirs(self._tempdir))
            for existing_top_dir in utils.get_top_level_dirs(dest_path):
                if existing_top_dir in generated_top_dirs:
                    raise ValueError(f"The directory with the name "
                                     f"{existing_top_dir} is already present in"
                                     f"{dest_path}.")

            copy_tree(self._tempdir, dest_path)
            return dest_path

        return self._tempdir

    def _initialize_local_directories(self, json_file_path: str):
        """
        Initializes `self._tempdir`, updates `self._json_paths` and
        `self._module_paths`.
        Args:
            json_file_path: Current json file being parsed.
        """
        self._tempdir = tempfile.mkdtemp()

        # Update the list of directories to be examined for imported configs
        self._update_json_paths(json_file_path)

        # Additional list of directories to be examined for python imports
        self._module_paths.extend([self._tempdir])

    def _update_json_paths(self, json_file_path):
        """
        Update `self._json_paths` with `json_file_path`.
        """
        self._json_paths.extend([
            os.path.dirname(os.path.realpath(json_file_path)), os.getcwd()])

    def _generate_ontology(self, json_file_path: str,
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
            rec_visited_paths = defaultdict(lambda: False)

        visited_paths[json_file_path] = True
        rec_visited_paths[json_file_path] = True

        self._update_json_paths(json_file_path)

        with open(json_file_path, 'r') as f:
            curr_str = f.read()
        curr_dict = json.loads(curr_str)

        # Extract imported json files and generate ontology for them.
        json_imports: List[str] = curr_dict[self._IMPORTS] \
            if self._IMPORTS in curr_dict else []

        modules_to_import: List[str] = []
        for imported in json_imports:
            resolved_imported = utils.search_in_dirs(imported, self._json_paths)
            if resolved_imported is None:
                raise ValueError(f"Ontology corresponding to {imported} not"
                                 f"found in the current directory or the "
                                 f"directory of original json config.")
            else:
                imported = resolved_imported
            if imported in rec_visited_paths:
                raise ValueError(f"Ontology corresponding to {imported} already"
                                 f" generated, cycles not permitted, aborting")
            elif imported not in visited_paths:
                modules_to_import.extend(self._generate_ontology(
                    imported, visited_paths, rec_visited_paths))

        # once the ontologies for all the imported files is generated, generate
        # ontology of the current file
        modules_to_import = self._generate_ontology_per_ontology_dict(
            curr_dict, modules_to_import)
        rec_visited_paths[json_file_path] = False
        return modules_to_import

    def _generate_ontology_per_ontology_dict(self, onto_dict: Dict,
                                             modules_to_import: List[str]) \
            -> List[str]:
        """
        Generates ontology code for ontology dictionary extracted from a json
        config. Appends entry code to the corresponding module. Creates a new
        module file if module is generated for the first time.
        Args:
            onto_dict: Ontology dictionary extracted from a json config.
            modules_to_import: Dependencies to be imported by generated modules.

        Returns:
            Modules to be imported by dependencies of the current ontology.
        """
        entry_definitions: List[Dict] = onto_dict[self._ENTRY_DEFINITIONS]

        new_modules_to_import = []
        for entry_definition in entry_definitions:
            entry_pkg, entry_filename, entry_name, entry_code, attributes \
                = self._get_entry_code(entry_definition)
            globals_code = f'__all__.extend("{entry_name}")'
            entry_code = '\n\n\n'.join([globals_code, entry_code]) + '\n\n'

            entry_module_name: str = f"{entry_pkg}.{entry_filename}"
            entry_class_name: str = f"{entry_module_name}.{entry_name}"

            try:
                # Creating entry directory and file in the tempdir if required.
                entry_dir: str = os.path.join(self._tempdir,
                                              *entry_pkg.split('.'))
                Path(entry_dir).mkdir(parents=True, exist_ok=True)

                entry_file: str = \
                    f'{os.path.join(entry_dir, entry_filename)}.py'

                # Creating the file if it does not exist.
                if not os.path.exists(entry_file):
                    ontology_file_docstring = '\n'.join(self.ignore_errors + [
                        '"""',
                        'Automatically generated file. Do not change by hand',
                        '"""'])
                    all_imports: Set[str] = set()
                    import_code = ''
                    for import_ in self.required_imports + modules_to_import:
                        if import_ not in all_imports:
                            import_code += f"import {import_}\n"
                        all_imports.add(import_)

                    with open(entry_file, 'w') as f:
                        f.write('\n\n'.join([ontology_file_docstring,
                                             import_code,
                                             '__all__ = []', '']))

                # Validating and adding entry class to the module.
                if entry_name in utils.get_user_objects_from_module(
                        entry_module_name, self._module_paths):
                    raise ValueError(f"The entry class with name {entry_name} "
                                     f"is already present in the module "
                                     f"{entry_module_name}")

                if entry_class_name in self._allowed_types_tree:
                    raise ValueError(f"The class named {entry_class_name} is"
                                     f"already declared")

                with open(entry_file, 'a') as f:
                    f.write(entry_code)

            except ValueError:
                self.cleanup_generated_ontology()
                raise

            # Modules to be imported by the dependencies.
            new_modules_to_import.append(entry_module_name)

            # Adding entry and the corresponding attributes to the allowed types
            # for validation.
            self._allowed_types_tree[entry_class_name] = set()
            for attribute_name in attributes:
                if attribute_name in self._allowed_types_tree[entry_class_name]:
                    raise Warning(
                        f"Attribute type for the entry {entry_class_name} and "
                        f"the attribute {attribute_name} already present in "
                        f"the ontology, will be overridden")
                self._allowed_types_tree[entry_class_name].add(attribute_name)

        return new_modules_to_import

    def cleanup_generated_ontology(self):
        """
        Deletes the generated ontology files.
        """
        shutil.rmtree(self._tempdir)

    def _get_entry_code(self, entry_definition: Dict) \
            -> Tuple[str, str, str, str, List[str]]:
        """
        Args:
            entry_definition: Dictionary containing specifications for an entry.

        Returns: extracted entry information: entry package string, entry
        filename, entry class name, generated entry code and entry attribute
        names.
        """
        # reading the entry definition dictionary
        entry_full_name: str = entry_definition[self._ENTRY_NAME]
        entry_package, entry_file, entry_name = entry_full_name.rsplit('.', 2)

        parent_entry: str = entry_definition[self._PARENT_ENTRY]

        # validate if the entry parent is present in the tree
        if parent_entry not in self._allowed_types_tree:
            raise ValueError(f"Cannot add {entry_full_name} to the ontology as "
                             f"it's parent entry {parent_entry} is not present "
                             f"in the ontology.")

        if entry_full_name in self._allowed_types_tree:
            raise Warning(f"Entry {entry_full_name} already present in the "
                          f"ontology, will be overridden.")

        # getting entry descriptions
        entry_desc = entry_definition[self._ENTRY_DESC] \
            if self._ENTRY_DESC in entry_definition and \
            entry_definition[self._ENTRY_DESC] is not None and \
            entry_definition[self._ENTRY_DESC].strip() != '' else None

        # Generate arguments to be passed inside `self.__init__` and
        # `self.__super__` according to whether the entry is a descendants of
        # one of the base entries defined in `base_ontology_module`.
        base_entry: str = self._get_and_set_base_entry(entry_full_name,
                                                       parent_entry)
        init_args: str = self._top_entries[base_entry]
        super_args: str = ', '.join([item.split(':')[0].strip()
                                     for item in init_args.split(',')])

        # generate code inside init function
        class_line: str = f"class {entry_name}({parent_entry}):"
        attributes, attribute_code, attribute_desc_map, attribute_type_map = \
            self._get_attribute_code_description_type(entry_definition,
                                                      entry_full_name)
        class_desc, init_desc = self._generate_class_description(
            entry_desc, attributes, attribute_desc_map, attribute_type_map)

        class_desc_lst = [class_desc] if class_desc is not None else []
        init_desc_lst = [init_desc] if init_desc is not None else []

        init_line: str = f"{self._INDENT}def __init__(self, {init_args}):"
        super_line: str = f"{2 * self._INDENT}super().__init__({super_args})"

        entry_code_lines = [class_line] + class_desc_lst + [init_line] + \
            init_desc_lst + [super_line, attribute_code]

        entry_code = '\n'.join(entry_code_lines)

        return entry_package, entry_file, entry_name, entry_code, attributes

    def _get_and_set_base_entry(self, entry_name: str, parent_entry: str) \
            -> str:
        """
        Function to return `base_entry` which is the entry on which the
         arguments would be based and populates `self._base_entry_map_seen`
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
        if parent_entry in self._top_entries:
            base_entry: str = parent_entry
        else:
            base_entry = self._base_entry_map_seen[parent_entry]
        self._base_entry_map_seen[entry_name] = base_entry
        return base_entry

    def _get_attribute_code_description_type(self, entry_definition: Dict,
                                             entry_full_name: str) \
            -> Tuple[List[str], str, Dict[str, Optional[str]], Dict[str, str]]:
        """
        Parses entry definition for a given entry and extracts it's attributes.
        Args:
            entry_definition: User-provided entry definition dictionary.
            entry_full_name: Full class name of the entry.

        Returns:
            Details of the attributes extracted from `entry_definition`:
            Ordered list of attribute names, generated attribute code, mapping
            from attribute name to attribute description string, mapping from
            attribute name to attribute type string.
        """
        attributes_list = []
        attribute_code = ''
        attribute_desc_map, attribute_type_map = dict(), dict()
        # extracting attributes
        attributes: List[Dict] = entry_definition[self._ATTRIBUTES] \
            if self._ATTRIBUTES in entry_definition else []

        for attribute in attributes:
            attribute_name, attribute_init_code, attribute_type \
                = self._get_attribute_init_code(attribute, entry_full_name)
            attribute_code += f"{attribute_init_code}"
            attributes_list.append(attribute_name)
            attribute_type_map[attribute_name] = attribute_type
            att_desc = attribute[self._ATTRIBUTE_DESC] \
                if self._ATTRIBUTE_DESC in attribute else None
            attribute_desc_map[attribute_name] = None \
                if att_desc is None or att_desc.strip() == '' else att_desc

        # generate getter and setter functions
        for attribute in attributes:
            attribute_code += '\n'
            attribute_getter_setter_code: str = \
                self._get_attribute_getter_setter_code(attribute)
            attribute_code += f"{attribute_getter_setter_code}"

        return (attributes_list, attribute_code, attribute_desc_map,
                attribute_type_map)

    def _get_attribute_init_code(self, attribute: Dict, entry_full_name: str)\
            -> Tuple[str, str, str]:
        """
        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: Code generated by an attribute in the `self.__init__` function.
        """
        attribute_name: str = attribute[self._ATTRIBUTE_NAME]
        attribute_type: str = attribute[self._ATTRIBUTE_TYPE]
        if attribute_type in self._COMPOSITE_TYPES:
            if self._ELEMENT_TYPE not in attribute:
                raise ValueError(f"Element type of the composite "
                                 f"{attribute_type} should be indicated with "
                                 f"the key `element_type`")
            element_type = attribute[self._ELEMENT_TYPE]
        else:
            element_type = None

        # attribute type should be present in the validation tree
        if attribute_type not in self._allowed_types_tree:
            raise ValueError(f"Attribute type for the entry {entry_full_name}"
                             f"and the attribute {attribute_name} not declared"
                             f"in the ontology")

        # element type should be present in the validation tree
        if attribute_type in self._COMPOSITE_TYPES:
            if not (element_type in self._allowed_types_tree
                    or element_type == entry_full_name):
                raise ValueError(f"Element type {element_type} for the entry "
                                 f"{entry_full_name} and the attribute "
                                 f"{attribute_name} not declared in ontology")
            attribute_type = f"typing.{attribute_type}"

        if self._ATTRIBUTE_DEFAULT_VALUE in attribute and \
                attribute[self._ATTRIBUTE_DEFAULT_VALUE] is not None:
            attribute_default: str = attribute[self._ATTRIBUTE_DEFAULT_VALUE]
            if attribute_type == "str":
                attribute_default = f"'{attribute_default}'"
        else:
            attribute_default = "None"

        composite_code: str = f"[{element_type}]" \
            if element_type is not None else ''

        code_string: str = f"{self._INDENT}{self._INDENT}" \
            f"self._{attribute_name}: typing.Optional[{attribute_type}" \
            f"{composite_code}] = {attribute_default}\n"

        return attribute_name, code_string, attribute_type

    def _get_attribute_getter_setter_code(self, attribute: Dict) -> str:
        """

        Args:
            attribute: Dictionary corresponding to a single attribute for an
            entry.

        Returns: getter and setter functions generated by an attribute.
        """
        attribute_name: str = attribute[self._ATTRIBUTE_NAME]
        attribute_type: str = attribute[self._ATTRIBUTE_TYPE]

        if attribute_type in self._COMPOSITE_TYPES:
            attribute_type = f'typing.{attribute_type}'
            if self._ELEMENT_TYPE in attribute:
                attribute_type += f'[{attribute[self._ELEMENT_TYPE]}]'

        return f"{self._INDENT}@property\n" \
            f"{self._INDENT}def {attribute_name}(self):\n" \
            f"{self._INDENT}{self._INDENT}return self._{attribute_name}\n\n" \
            f"{self._INDENT}def set_{attribute_name}" \
            f"(self, {attribute_name}: {attribute_type}):\n" \
            f"{self._INDENT}{self._INDENT}self.set_fields(" \
            f"_{attribute_name}={attribute_name})\n"

    def _generate_class_description(self, entry_desc: Optional[str],
                                    attributes: List[str],
                                    attributes_desc: Dict[str, Optional[str]],
                                    attribute_types: Dict[str, str])\
            -> Tuple[Optional[str], Optional[str]]:
        """
        Generate class and attribute description code for an entry.
        Args:
            entry_desc: User-provided entry description.
            attributes: List of attribute names to specify the order of
            generated descriptions.
            attributes_desc: Mapping from attribute name to its string
            description.
            attribute_types: Mapping from attribute name to their type to be
            added in attribute description.

        Returns:
            Class (entry) description and attribute (`__init__` argument)
            descriptions.
        """
        all_att_none = all([attributes_desc[att] is None for att in attributes])
        if entry_desc is None or entry_desc.strip() == '':
            entry_desc = None
        else:
            entry_desc = self._INDENT + ('\n' + self._INDENT).join(
                ['"""', entry_desc, '"""'])

        if all_att_none:
            init_desc = None
        else:
            arg_desc_lines = [f"{attribute} ({attribute_types[attribute]}): "
                              f"{attributes_desc[attribute]}"
                              for attribute in attributes]
            arg_desc = self._INDENT + \
                ('\n' + 3 * self._INDENT).join(arg_desc_lines)

            init_desc = 2 * self._INDENT + \
                ('\n' + 2 * self._INDENT).join(['"""', 'Attributes:', arg_desc,
                                                '"""'])
        return entry_desc, init_desc
