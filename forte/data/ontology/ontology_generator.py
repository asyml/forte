"""
    Module to automatically generate python ontology given json file
"""
import json
import os
import shutil

from typing import Dict, Tuple, List


class OntologyGenerator:
    """
    Class to generate a python ontology file given ontology in json format
    Example:
        >>> _, generated_file, _ = OntologyGenerator().generate_ontology(
        'test/example_ontology.json')
        >>> assert open(generated_file, 'r').read() == \
                   open('test/true_example_ontology.py', 'r').read()
    """

    # string constants
    _ONTOLOGY_NAME = "ontology_name"
    _IMPORTS = "imports"
    _ENTRY_DEFINITIONS = "entry_definitions"
    _ENTRY_NAME = "entry_name"
    _PARENT_ENTRY = "parent_entry"
    _ATTRIBUTES = "attributes"
    _ATTRIBUTE_NAME = "attribute_name"
    _ATTRIBUTE_TYPE = "attribute_type"
    _ATTRIBUTE_DEFAULT_VALUE = "attribute_default_value"
    _INDENT = ' ' * 4

    # allowed parent entries and init arguments
    _INIT_ARGUMENTS = {"forte.data.ontology.top.Annotation":
                       "begin: int, end: int",

                       "forte.data.ontology.top.Link":
                       "parent: Optional[Entry] = None, "
                       "child: Optional[Entry] = None",

                       "forte.data.ontology.top.Group":
                       "members: Optional[Set[Entry]] = None"}

    def __init__(self):
        # mapping from ontology full name to corresponding generated file and
        # folder path
        self._generated_ontology_record: Dict[str, Tuple[str, str]] = {}
        # mapping from entries seen till now with their "base parents" which
        # should be one of the keys of self._INIT_ARGUMENTS
        self._base_entry_map_seen: Dict[str, str] = {}

    def generate_ontology(self, json_file_path):
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
            ontology: dict = json.loads(json_string)
            ontology_full_name: str = ontology[self._ONTOLOGY_NAME]
            imports: List[str] = ontology[self._IMPORTS]
            entry_definitions: List[str] = ontology[self._ENTRY_DEFINITIONS]

            # creating ontology code
            ontology_file_docstring = '"""\nOntology file for ' \
                                      'forte.data.ontology.example_ontology\n' \
                                      'Automatically generated file. ' \
                                      'Do not change by hand\n"""'

            custom_imports = f"import typing\n"

            ontology_name_code, ontology_folder, ontology_file_name \
                = self.get_ontology_name_code(ontology_full_name)

            ontology_code = f"{ontology_file_docstring}\n" \
                            f"{custom_imports}" \
                            f"{ontology_name_code}"

            for import_module in imports:
                ontology_code += f"import {import_module}\n"

            for entry_definition in entry_definitions:
                ontology_code += self.get_entry_code(entry_definition)

            # creating ontology package directories in the current directory if
            # required
            ontology_dir_path = os.path.join(os.getcwd(), ontology_folder)
            if not os.path.isdir(ontology_dir_path):
                os.makedirs(ontology_dir_path)

            # creating the ontology python file and populating it
            ontology_file_path = os.path.join(ontology_dir_path,
                                              ontology_file_name)

            with open(ontology_file_path, 'w') as file:
                file.write(ontology_code)

            # recording the generate ontology file details
            self._generated_ontology_record[ontology_full_name] \
                = (ontology_file_path, ontology_folder)

            return ontology_full_name, ontology_file_path, ontology_folder

    def cleanup_generated_ontology(self, ontology_full_name):
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
            top_path = os.path.join(os.getcwd(), ontology_folder.split('/')[0])
            all_files = [files for _, _, files in os.walk(top_path)]
            num_files = sum([len(files) for files in all_files])
            if num_files == 0:
                shutil.rmtree(top_path)

    def get_ontology_name_code(self, ontology_full_name):
        """
        Function to parse ontology package and name, create required folders
        if the ontology package is not the same as current package, and generate
        corresponding code
        Args:
            ontology_full_name: Full namespace of the ontology

        Returns:
            - The generated code
            - Directory path where the generated file would be stored
            - Name of the file to be generated
        """
        namespace = ontology_full_name.split('.')
        ontology_package = '.'.join(namespace[0:-1])
        code_str = f"import {ontology_package}\n"

        ontology_folder = os.path.join(*namespace[0:-1])
        ontology_file_name = namespace[-1] + '.py'

        return code_str, ontology_folder, ontology_file_name

    def get_entry_code(self, entry_definition):
        """
        Args:
            entry_definition: entry definition dictionary

        Returns: code generated by an entry definition
        """

        entry_full_name = entry_definition[self._ENTRY_NAME]
        entry_name = entry_full_name.split('.')[-1]
        parent_entry = entry_definition[self._PARENT_ENTRY]

        attributes = entry_definition[self._ATTRIBUTES] \
            if self._ATTRIBUTES in entry_definition else []

        base_entry = self.get_and_set_base_entry(entry_full_name, parent_entry)
        init_arguments = self._INIT_ARGUMENTS[base_entry]
        super_arguments = ', '.join([item.split(':')[0].strip()
                                     for item in init_arguments.split(',')])

        class_line = f"class {entry_name}({parent_entry}):"
        init_line = f"def __init__(self, {init_arguments}):"
        super_line = f"super().__init__({super_arguments})\n"

        entry_code = f"\n\n{class_line}\n{self._INDENT}{init_line}\n" \
            f"{self._INDENT}{self._INDENT}{super_line}"

        for attribute in attributes:
            attribute_init_code = self.get_attribute_init_code(attribute)
            entry_code += f"{attribute_init_code}"

        for attribute in attributes:
            entry_code += '\n'
            attribute_getter_setter_code = \
                self.get_attribute_getter_setter_code(attribute)
            entry_code += f"{attribute_getter_setter_code}"

        return entry_code

    def get_and_set_base_entry(self, entry_name, parent_entry):
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
            base_entry = self._base_entry_map_seen[parent_entry]
        self._base_entry_map_seen[entry_name] = base_entry
        return base_entry

    def get_attribute_init_code(self, attribute):
        """
        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: code generated by an attribute in the self.__init__ function

        """
        attribute_name = attribute[self._ATTRIBUTE_NAME]
        attribute_type = attribute[self._ATTRIBUTE_TYPE]

        if self._ATTRIBUTE_DEFAULT_VALUE in attribute and \
                attribute[self._ATTRIBUTE_DEFAULT_VALUE] is not None:
            attribute_default_value = attribute[self._ATTRIBUTE_DEFAULT_VALUE]
            if attribute_type == "str":
                attribute_default_value = f"'{attribute_default_value}'"
        else:
            attribute_default_value = "None"

        code_string = f"{self._INDENT}{self._INDENT}self._{attribute_name}: " \
            f"typing.Optional[{attribute_type}] = {attribute_default_value}\n"

        return code_string

    def get_attribute_getter_setter_code(self, attribute):
        """

        Args:
            attribute: Dictionary corresponding to a single attribute
            for an entry

        Returns: getter and setter functions generated by an attribute

        """
        attribute_name = attribute[self._ATTRIBUTE_NAME]
        attribute_type = attribute[self._ATTRIBUTE_TYPE]
        return f"{self._INDENT}@property\n" \
            f"{self._INDENT}def {attribute_name}(self):\n" \
            f"{self._INDENT}{self._INDENT}return self._{attribute_name}\n\n" \
            f"{self._INDENT}def set_{attribute_name}" \
            f"(self, {attribute_name}: {attribute_type}):\n" \
            f"{self._INDENT}{self._INDENT}self.set_fields(" \
            f"_{attribute_name}={attribute_name})\n"
