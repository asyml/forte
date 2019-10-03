"""
    Module to automatically generate python ontology given json file
    Performs a preliminary check of class dependencies
"""
import json
import os
import shutil

from typing import Dict, Tuple, List, Set

from forte.data.ontology import utils


class OntologyCodeGenerator:
    """
    Class to generate a python ontology file given ontology in json format
    Example:
        >>> _, generated_file, _ = OntologyCodeGenerator().generate_ontology(
        'test/example_ontology_config.json')
        >>> assert open(generated_file, 'r').read() == \
                   open('test/true_example_ontology.py', 'r').read()
    """

    # string constants
    _ONTOLOGY_NAME: str = "ontology_name"
    _IMPORTS: str = "imports"
    _ENTRY_DEFINITIONS: str = "entry_definitions"
    _ENTRY_NAME: str = "entry_name"
    _PARENT_ENTRY: str = "parent_entry"
    _ATTRIBUTES: str = "attributes"
    _ATTRIBUTE_NAME: str = "attribute_name"
    _ATTRIBUTE_TYPE: str = "attribute_type"
    _ELEMENT_TYPE: str = "element_type"
    _ATTRIBUTE_DEFAULT_VALUE: str = "attribute_default_value"
    _INDENT: str = ' ' * 4

    # allowed parent entries and init arguments
    _INIT_ARGUMENTS: Dict[str, str] = {
        "forte.data.ontology.top.Annotation":
            "pack:forte.data.data_pack.DataPack, "
            "begin: int, "
            "end: int",
        "forte.data.ontology.top.Link":
            "pack:data_pack.DataPack, "
            "parent: Optional[Entry] = None, "
            "child: Optional[Entry] = None",
        "forte.data.ontology.top.Group":
            "pack:data_pack.DataPack, "
            "members: Optional[Set[Entry]] = None"
         }

    _PRIMARY_TYPES = ["bool", "int", "float", "str"]
    _COMPOSITE_TYPES = ["List"]

    def __init__(self):
        # imports to be added to the ontology other than the ones defined in the
        # json
        self.additional_imports: List[str] = ['typing', 'forte.data.data_pack']
        # mapping from ontology full name to corresponding generated file and
        # folder path
        self._generated_ontology_record: Dict[str, Tuple[str, str]] = {}
        # mapping from entries seen till now with their "base parents" which
        # should be one of the keys of self._INIT_ARGUMENTS
        self._base_entry_map_seen: Dict[str, str] = {}
        # declare a tree to store the entries and their attributes in order to
        # validate the class dependency structure
        self._validation_tree: Dict[str, Set] = {}
        self._initialize_primary_types()

    def _initialize_primary_types(self):
        """
            Initialize self._validation_tree with primary python data types
        """
        self._validation_tree = dict()
        for allowed_type in self._PRIMARY_TYPES + self._COMPOSITE_TYPES:
            self._validation_tree[allowed_type] = set()

    def generate_ontology(self, json_file_path: str):
        """
        Function to generate and save the python ontology code after reading
        ontology from the input json file
        Args:
            json_file_path: The input json file to read the ontology from
        Returns:
            - Ontology module name
            - Path of python ontology file if the code executes correctly
            - Top level path of nested packages created
        """
        with open(json_file_path, 'r') as f:
            # reading ontology
            json_string: str = f.read()
            ontology: Dict = json.loads(json_string)
            ontology_full_name: str = ontology[self._ONTOLOGY_NAME]
            custom_imports: List[str] = ontology[self._IMPORTS]
            entry_definitions: List[Dict] = ontology[self._ENTRY_DEFINITIONS]

            all_imports = self.additional_imports + custom_imports

            # validation of imported classes
            package_imports = all_imports[:]
            for import_module in custom_imports:
                for class_str in utils.get_classes_from_module(import_module):
                    package_imports.append(f"{import_module}.{class_str}")

            for import_module in package_imports:
                if import_module in self._validation_tree:
                    raise Warning(f"The module {import_module} is already"
                                  f"added in the ontology, will be overridden")
                self._validation_tree[import_module] = set()

            # getting ontology info
            ontology_folder, ontology_file_name \
                = self._get_ontology_info(ontology_full_name)

            ontology_file_docstring: str = f'# mypy: ignore-errors\n' \
                                           f'"""\nOntology file for ' \
                                           f'{ontology_full_name}\n' \
                                           f'Automatically generated file. ' \
                                           f'Do not change by hand\n"""'

            import_code = ''
            for import_module in all_imports:
                import_code += f"import {import_module}\n"

            ontology_code: str = f"{ontology_file_docstring}\n" \
                                 f"{import_code}"

            for entry_definition in entry_definitions:
                ontology_code += self._get_entry_code(entry_definition)

            # creating ontology package directories in the current directory if
            # required
            ontology_dir_path: str = os.path.join(os.getcwd(), ontology_folder)

            # validating and adding ontology_full_name
            if os.path.isdir(ontology_dir_path) and ontology_full_name in \
                    utils.get_classes_from_folder(ontology_dir_path):
                raise ValueError(f"The class named {ontology_full_name} is"
                                 f"already present in the {ontology_dir_path}")

            if ontology_full_name in self._validation_tree:
                raise ValueError(f"The class named {ontology_full_name} is"
                                 f"already present")

            self._validation_tree[ontology_full_name] = set()

            # Clean up in case of any error
            try:
                if not os.path.isdir(ontology_dir_path):
                    os.makedirs(ontology_dir_path)

                # creating the ontology python file and populating it
                ontology_file_path: str = os.path.join(ontology_dir_path,
                                                       ontology_file_name)

                with open(ontology_file_path, 'w') as file:
                    file.write(ontology_code)

                # recording the generate ontology file details
                self._generated_ontology_record[ontology_full_name] \
                    = (ontology_file_path, ontology_folder)

            except Exception:
                self.cleanup_generated_ontology(ontology_full_name)
                raise

            return ontology_full_name, ontology_file_path, ontology_folder

    def cleanup_generated_ontology(self, ontology_full_name: str):
        """
        Deletes the generated ontology file and corresponding folder if empty
        Args:
            ontology_full_name: Full name of the ontology (with package name)
        """
        if ontology_full_name in self._generated_ontology_record:
            ontology_file_path, ontology_folder = \
                self._generated_ontology_record[ontology_full_name]

            # delete generated file
            os.remove(ontology_file_path)

            # delete generated folders
            top_path: str = os.path.join(os.getcwd(),
                                         ontology_folder.split('/')[0])
            all_files = [files for _, _, files in os.walk(top_path)]
            num_files = sum([len(files) for files in all_files])
            if num_files == 0:
                shutil.rmtree(top_path)

    @staticmethod
    def _get_ontology_info(ontology_full_name: str):
        """
        Function to parse ontology package and name, create required folders
        if the ontology package is not the same as current package
        Args:
            ontology_full_name: Full namespace of the ontology

        Returns:
            - Directory path where the generated file would be stored
            - Name of the file to be generated
        """
        namespace: List[str] = ontology_full_name.split('.')

        ontology_folder: str = os.path.join(*namespace[0:-1])
        ontology_file_name: str = namespace[-1] + '.py'

        return ontology_folder, ontology_file_name

    def _get_entry_code(self, entry_definition: Dict):
        """
        Args:
            entry_definition: entry definition dictionary

        Returns: code generated by an entry definition
        """

        # reading the entry definition dictionary
        entry_full_name: str = entry_definition[self._ENTRY_NAME]
        entry_split = entry_full_name.split('.')
        entry_name: str = entry_split[-1]

        parent_entry: str = entry_definition[self._PARENT_ENTRY]

        # validate if the entry parent is present in the tree
        if parent_entry not in self._validation_tree:
            raise ValueError(f"Cannot add {entry_full_name} to the ontology as "
                             f"it's parent entry {parent_entry} is not present "
                             f"in the ontology.")

        if entry_full_name in self._validation_tree:
            raise Warning(f"Entry {entry_full_name} already present in the "
                          f"ontology, will be overridden.")

        # add the entry to the tree
        self._validation_tree[entry_full_name] = set()

        # generate arguments to be passed inside init and super functions
        # according to whether the entry is a descended of Annotation, Link
        # or Group
        base_entry: str = self._get_and_set_base_entry(entry_full_name,
                                                       parent_entry)
        init_arguments: str = self._INIT_ARGUMENTS[base_entry]
        super_arguments: str = ', '.join(
            [item.split(':')[0].strip() for item in init_arguments.split(',')])

        # generate code inside init function
        class_line: str = f"class {entry_name}({parent_entry}):"
        init_line: str = f"def __init__(self, {init_arguments}):"
        super_line: str = f"super().__init__({super_arguments})\n"

        entry_code: str = f"\n\n{class_line}\n{self._INDENT}{init_line}\n"\
            f"{self._INDENT}{self._INDENT}{super_line}"

        entry_code += self._get_attribute_code(entry_definition,
                                               entry_full_name)
        return entry_code

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
            `DependencyToken` inherits `Token` inherits `Annotation`
            The base_entry for both `DependencyToken` and `Token` should be
            `Annotation`
        """
        if parent_entry in self._INIT_ARGUMENTS:
            return parent_entry
        else:
            base_entry: str = self._base_entry_map_seen[parent_entry]
        self._base_entry_map_seen[entry_name] = base_entry
        return base_entry

    def _get_attribute_code(self, entry_definition, entry_full_name):
        attribute_code = ''
        # extracting attributes
        attributes: List[Dict] = entry_definition[self._ATTRIBUTES] \
            if self._ATTRIBUTES in entry_definition else []

        for attribute in attributes:
            attribute_init_code: str = self._get_attribute_init_code(
                attribute, entry_full_name)
            attribute_code += f"{attribute_init_code}"

        # generate getter and setter functions
        for attribute in attributes:
            attribute_code += '\n'
            attribute_getter_setter_code: str = \
                self._get_attribute_getter_setter_code(attribute)
            attribute_code += f"{attribute_getter_setter_code}"
        return attribute_code

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
            if element_type not in self._validation_tree:
                raise ValueError(f"Element type for the entry {entry_full_name}"
                                 f"and the attribute {attribute_name} not "
                                 f"declared in the ontology")
            attribute_type = f"typing.{attribute_type}"

        # adding attribute to the entry
        if attribute_name in self._validation_tree[entry_full_name]:
            raise Warning(f"Attribute type for the entry {entry_full_name} and "
                          f"the attribute {attribute_name} already present in "
                          f"the ontology, will be overridden")
        self._validation_tree[entry_full_name].add(attribute_name)

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

        return code_string

    def _get_attribute_getter_setter_code(self, attribute: Dict):
        """

        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: getter and setter functions generated by an attribute

        """
        attribute_name: str = attribute[self._ATTRIBUTE_NAME]
        attribute_type: str = attribute[self._ATTRIBUTE_TYPE]

        return f"{self._INDENT}@property\n" \
            f"{self._INDENT}def {attribute_name}(self):\n" \
            f"{self._INDENT}{self._INDENT}return self._{attribute_name}\n\n" \
            f"{self._INDENT}def set_{attribute_name}" \
            f"(self, {attribute_name}: {attribute_type}):\n" \
            f"{self._INDENT}{self._INDENT}self.set_fields(" \
            f"_{attribute_name}={attribute_name})\n"
