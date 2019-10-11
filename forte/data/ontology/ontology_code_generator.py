"""
    Module to automatically generate python ontology given json file
    Performs a preliminary check of class dependencies
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
from typing import Dict, Tuple, List, Set, Optional

import typed_ast.ast3 as ast
import typed_astunparse as ast_unparse

from forte.data.ontology import utils
from forte.data.ontology.base import top


class OntologyCodeGenerator:
    """
    Class to generate a python ontology file given ontology in json format
    Example:
        >>> _, generated_file, _ = OntologyCodeGenerator().generate_ontology(
        'test/example_ontology_config.json')
        >>> assert open(generated_file, 'r').read() == \
                   open(example_ontology.py).read()
    """

    # string constants
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
    _INDENT: str = ' ' * 4

    _PRIMARY_TYPES = ["bool", "int", "float", "str"]
    _COMPOSITE_TYPES = ["List"]

    def __init__(self, ontology_base_module: ModuleType = top):
        self.ontology_base_module = ontology_base_module
        # modules to be added to the ontology other than the ones defined in the
        # json, in the order of preference of modules
        self.default_imports: List[str] = ['typing', 'forte.data.data_pack',
                                           ontology_base_module.__name__]
        # lines to be added in the beginning of the file to pass errors
        self.ignore_errors: List[str] = [
            '# flake8: noqa', '# mypy: ignore-errors',
            '# pylint: disable=line-too-long,trailing-newlines']
        # mapping from package name to corresponding generated file and
        # folder path
        self._generated_ontology_record: Dict[str, Tuple[str, str]] = {}
        # entries corresponding to top.py
        self._top_entries: Dict[str, str] = {}
        self._initialize_top_entries()
        # mapping from entries seen till now with their "base parents" which
        # should be one of the keys of self._top_entries
        self._base_entry_map_seen: Dict[str, str] = {}
        # declare a tree to store the entries and their attributes in order to
        # validate the class dependency structure
        self._validation_tree: Dict[str, Set] = {}
        self._initialize_primary_types()
        # create a `tempdir` where the generated files will be saved temporarily
        # until the generation is complete
        self._tempdir: Optional[str] = None
        self._dirs: Optional[List[str]] = None

    def _initialize_primary_types(self):
        """
            Initialize self._validation_tree with primary python data types
        """
        for allowed_type in self._PRIMARY_TYPES + self._COMPOSITE_TYPES:
            self._validation_tree[allowed_type] = set()

    def _initialize_top_entries(self):
        """
        Parses the file corresponding to `self.ontology_base_module` and
        extracts the class definitions and the corresponding `__init__` args.
        Returns: Dictionary mapping class definitions with `__init__` arguments.
        """
        tree = ast.parse(open(self.ontology_base_module.__file__).read())
        imports = {}
        base_entries = {}

        # adding all the imports
        for elem in tree.body:
            if isinstance(elem, ast.Import):
                for import_ in elem.names:
                    import_name = import_.name.split('.')[-1]
                    imports[import_name] = import_.name

            if isinstance(elem, ast.ImportFrom):
                for import_ in elem.names:
                    imports[import_.name] = f"{elem.module}.{import_.name}"

            # adding all the module objects defined in __all__ to imports
            if isinstance(elem, ast.Assign) and len(elem.targets) > 0:
                if elem.targets[0].id == '__all__':
                    imports.update(
                        [(name.s,
                          f"{self.ontology_base_module.__name__}.{name.s}")
                         for name in elem.value.elts])

            # adding init arguments for each class
            if isinstance(elem, ast.ClassDef):
                for func in elem.body:
                    if isinstance(func, ast.FunctionDef) and \
                            func.name == '__init__':
                        for i, arg in enumerate(func.args.args):
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
                            func.args.args[i] = arg
                        args = ast_unparse.unparse(func.args).split(',', 1)
                        args_str = args[1].strip().replace('\n', '')
                        args_str = args_str.replace('  ', '')
                        base_entries[imports[elem.name]] = args_str
        self._top_entries = base_entries

    def generate_ontology(self, json_file_path: str,
                          destination_dir: Optional[str] = None,
                          is_dry_run: bool = False):
        """
            Function to generate and save the python ontology code after reading
            ontology from the input json file
            Args:
                json_file_path: The input json file to read the ontology from
                destination_dir: The folder in which config packages are to be
                generated. If not provided, current working directory is used.
                Not used if `is_dry_run` is `True`.
                is_dry_run: if `True`, creates the ontology in the temporary
                directory, else, creates the ontology in the `destination_dir`
            Returns:
                Base directory path in which the modules are created.
        """
        self._tempdir = tempfile.mkdtemp()
        # `tempdir`, current working directory and the directory of the
        # json_file_path to the list of directory trees to be examined
        self._dirs = [os.path.split(json_file_path)[0],
                      self._tempdir, os.getcwd()]

        sys.path = self._dirs + sys.path

        # validation of imported classes
        for import_module in self.default_imports:
            for obj_str in utils.get_user_objects_from_module(import_module):
                full_obj_str = f"{import_module}.{obj_str}"

                if full_obj_str in self._validation_tree:
                    raise Warning(f"The object {full_obj_str} is already "
                                  f"added in the ontology, will be overridden")
                self._validation_tree[full_obj_str] = set()

        self._generate_ontology(json_file_path,
                                visited_paths=defaultdict(lambda: False),
                                rec_visited_paths=defaultdict(lambda: False))

        # when everything is successfully completed, copy the contents of
        # `self._tempdir` to the provided folder
        if not is_dry_run:
            destination_dir = os.getcwd() if destination_dir is None else \
                destination_dir
            copy_tree(self._tempdir, destination_dir)
            return destination_dir

        return self._tempdir

    def _generate_ontology(self, json_file_path: str, visited_paths: Dict,
                           rec_visited_paths: Dict):

        visited_paths[json_file_path] = True
        rec_visited_paths[json_file_path] = True

        with open(json_file_path, 'r') as f:
            curr_str = f.read()

        curr_dict = json.loads(curr_str)

        # extract imported json files and generate ontology for them
        json_imports: List[str] = curr_dict[self._IMPORTS] \
            if self._IMPORTS in curr_dict else []
        modules_to_import: List[str] = []
        for imported in json_imports:
            resolved_imported = utils.search_in_dirs(imported, self._dirs)
            if resolved_imported is None:
                raise ValueError(f"Ontology corresponding to {imported} not"
                                 f"found in the current directory or the "
                                 f"directory of original json config")
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

    def _generate_ontology_per_ontology_dict(self, onto_dict,
                                             modules_to_import):
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
                # creating entry directory and file in the tempdir if required
                entry_dir: str = os.path.join(self._tempdir,
                                              *entry_pkg.split('.'))
                Path(entry_dir).mkdir(parents=True, exist_ok=True)

                # recording the generated entry file details
                entry_file: str = \
                    f'{os.path.join(entry_dir, entry_filename)}.py'
                self._generated_ontology_record[entry_class_name] = (entry_file,
                                                                     entry_dir)

                # creating the file if it does not exist
                if not os.path.exists(entry_file):
                    ontology_file_docstring = '\n'.join(self.ignore_errors +[
                        '"""',
                        'Automatically generated file. Do not change by hand',
                        '"""'])
                    all_imports = set()
                    import_code = ''
                    for import_ in self.default_imports + modules_to_import:
                        if import_ not in all_imports:
                            import_code += f"import {import_}\n"
                        all_imports.add(import_)

                    with open(entry_file, 'w') as f:
                        f.write('\n\n'.join([ontology_file_docstring,
                                             import_code,
                                             '__all__ = []', '']))

                # validating and adding entry class to the module
                if entry_name in utils.get_user_objects_from_module(
                        entry_module_name, self._dirs):
                    raise ValueError(f"The entry class with name {entry_name} "
                                     f"is already present in the module "
                                     f"{entry_module_name}")

                if entry_class_name in self._validation_tree:
                    raise ValueError(f"The class named {entry_class_name} is"
                                     f"already declared")

                with open(entry_file, 'a') as f:
                    f.write(entry_code)

            except ValueError:
                self.cleanup_generated_ontology(entry_class_name)
                raise

            # passing modules to be imported by child modules
            new_modules_to_import.append(entry_module_name)

            # adding entry and the corresponding attributes to validation tree
            self._validation_tree[entry_class_name] = set()
            for attribute_name in attributes:
                if attribute_name in self._validation_tree[entry_class_name]:
                    raise Warning(
                        f"Attribute type for the entry {entry_class_name} and "
                        f"the attribute {attribute_name} already present in "
                        f"the ontology, will be overridden")
                self._validation_tree[entry_class_name].add(attribute_name)
        return new_modules_to_import

    def cleanup_generated_ontology(self, class_name: str):
        """
        Deletes the generated ontology file and corresponding folder if empty
        Args:
            class_name: Full name of the ontology (with package name)
        """
        if class_name in self._generated_ontology_record:
            ontology_file_path, ontology_folder = \
                self._generated_ontology_record[class_name]

            # delete generated file
            os.remove(ontology_file_path)

            # delete generated folders
            top_path: str = os.path.join(os.getcwd(),
                                         ontology_folder.split('/')[0])
            all_files = [files for _, _, files in os.walk(top_path)]
            num_files = sum([len(files) for files in all_files])
            if num_files == 0:
                shutil.rmtree(top_path)

    def _get_entry_code(self, entry_definition: Dict):
        """
        Args:
            entry_definition: entry definition dictionary

        Returns: code generated by an entry definition
        """

        # reading the entry definition dictionary
        entry_full_name: str = entry_definition[self._ENTRY_NAME]
        entry_package, entry_file, entry_name = entry_full_name.rsplit('.', 2)

        parent_entry: str = entry_definition[self._PARENT_ENTRY]

        # validate if the entry parent is present in the tree
        if parent_entry not in self._validation_tree:
            raise ValueError(f"Cannot add {entry_full_name} to the ontology as "
                             f"it's parent entry {parent_entry} is not present "
                             f"in the ontology.")

        if entry_full_name in self._validation_tree:
            raise Warning(f"Entry {entry_full_name} already present in the "
                          f"ontology, will be overridden.")

        # getting entry descriptions
        entry_desc = entry_definition[self._ENTRY_DESC] \
            if self._ENTRY_DESC in entry_definition and \
            entry_definition[self._ENTRY_DESC] is not None and \
            entry_definition[self._ENTRY_DESC].strip() != '' else None

        # generate arguments to be passed inside init and super functions
        # according to whether the entry is a descended of Annotation, Link
        # or Group
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

    def _get_and_set_base_entry(self, entry_name: str, parent_entry: str):
        """
        Function to return `base_entry` which is the entry on which the
         arguments would be based and populates `self._base_entry_map_seen`
        Args:
            entry_name: the entry name for which the base_entry is
            to be returned
            parent_entry: parent of the entry name Note that the `base_entry` of
            the `entry_name` is same as the base entry of the `parent_entry`

        Returns:
            `base_entry` for the entry `entry_name`

        Example:
            If the subclass structure is -
            `Token` inherits `Token` inherits `Annotation`
            The base_entry for both `Token` and `Token` should be
            `Annotation`
        """
        if parent_entry in self._top_entries:
            base_entry: str = parent_entry
        else:
            base_entry = self._base_entry_map_seen[parent_entry]
        self._base_entry_map_seen[entry_name] = base_entry
        return base_entry

    def _get_attribute_code_description_type(self, entry_definition,
                                             entry_full_name):
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

    def _get_attribute_init_code(self, attribute: Dict, entry_full_name: str):
        """
        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: code generated by an attribute in the self.__init__ function

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
        if attribute_type not in self._validation_tree:
            raise ValueError(f"Attribute type for the entry {entry_full_name}"
                             f"and the attribute {attribute_name} not declared"
                             f"in the ontology")

        # element type should be present in the validation tree
        if attribute_type in self._COMPOSITE_TYPES:
            if not (element_type in self._validation_tree
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

    def _get_attribute_getter_setter_code(self, attribute: Dict):
        """

        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: getter and setter functions generated by an attribute

        """
        attribute_name: str = attribute[self._ATTRIBUTE_NAME]
        attribute_type: str = attribute[self._ATTRIBUTE_TYPE]
        if attribute_type in self._COMPOSITE_TYPES:
            attribute_type = f'typing.{attribute_type}'

        return f"{self._INDENT}@property\n" \
            f"{self._INDENT}def {attribute_name}(self):\n" \
            f"{self._INDENT}{self._INDENT}return self._{attribute_name}\n\n" \
            f"{self._INDENT}def set_{attribute_name}" \
            f"(self, {attribute_name}: {attribute_type}):\n" \
            f"{self._INDENT}{self._INDENT}self.set_fields(" \
            f"_{attribute_name}={attribute_name})\n"

    def _generate_class_description(self, entry_desc, attributes,
                                    attribute_descriptions, attribute_types):
        all_att_none = all([attribute_descriptions[att] is None for att in
                            attributes])
        if entry_desc is None or entry_desc.strip() == '':
            entry_desc = None
        else:
            entry_desc = self._INDENT + ('\n' + self._INDENT).join(
                ['"""', entry_desc, '"""'])

        if all_att_none:
            init_desc = None
        else:
            arg_desc_lines = [f"{attribute} ({attribute_types[attribute]}): "
                              f"{attribute_descriptions[attribute]}"
                              for attribute in attributes]
            arg_desc = self._INDENT + \
                ('\n' + 3 * self._INDENT).join(arg_desc_lines)

            init_desc = 2 * self._INDENT + \
                ('\n' + 2 * self._INDENT).join(['"""', 'Attributes:', arg_desc,
                                                '"""'])

        return entry_desc, init_desc
