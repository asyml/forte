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
import os
from abc import ABC
from typing import Optional, Any, List, Dict, Set
import itertools as it
from pathlib import Path
import pdb

from forte.data.ontology.utils import split_file_path


class ImportManager:
    r"""A naive implementation that records import strings and imported names
    to be used. Mainly used to avoid import name conflicts such as:
       -- from user.module import token
       -- from system.module import token
    """

    def __init__(self, module_name: str):
        self.__module_name = module_name
        self.__import_statements: List[str] = []
        self.__imported_names: Dict[str, str] = {}
        self.__short_name_pool: Set[str] = set()

    def is_valid_name(self, class_name):
        """
        Check whether the class name can be used.
        Args:
            class_name: The name to be check.

        Returns: True if the class_name can be used, which means it is either
            imported or it is of a primitive type.

        """
        return class_name in PrimitiveProperty.TYPES or self.is_imported(
            class_name)

    def is_imported(self, class_name):
        """
        Check whether this `class_name` is already imported here in the module.
        Args:
            class_name: The name to be checked for importing.

        Returns: True if the class_name is imported.

        """
        return class_name in self.__imported_names

    def get_name_to_use(self, full_name):
        return self.__imported_names[full_name]

    def get_import_statements(self):
        return sorted(self.__import_statements)

    def create_import_statement(self, full_name: str, as_name: str,
                                is_primitive: bool):
        if not is_primitive:
            parts = full_name.split('.')
            class_name = parts[-1]

            if len(parts) > 1:
                module_name = '.'.join(parts[:-1])
                if not module_name == self.__module_name:
                    # No need to import classes in the same module
                    if class_name == as_name:
                        import_statement = f'from {module_name} ' \
                                           f'import {class_name}'
                    else:
                        import_statement = f'from {module_name} ' \
                                           f'import {class_name} as {as_name}'

                    self.__import_statements.append(import_statement)
            else:
                if class_name == as_name:
                    import_statement = f'import {class_name}'
                else:
                    import_statement = f'import {class_name} as {as_name}'
                self.__import_statements.append(import_statement)

    def __find_next_available(self, class_name) -> str:
        counter = 0
        while True:
            as_name = f'{class_name}_{counter}'
            counter += 1
            if as_name not in self.__short_name_pool:
                break
        return as_name

    def __assign_as_name(self, full_name) -> str:
        class_name = full_name.split('.')[-1]
        if class_name not in self.__short_name_pool:
            self.__short_name_pool.add(class_name)
            return class_name
        else:
            as_name = self.__find_next_available(class_name)
            self.__short_name_pool.add(as_name)
            return as_name

    def add_object_to_import(self, full_name, is_primitive):
        if full_name is 'List':
            pdb.set_trace()

        if full_name not in self.__imported_names:
            if not is_primitive:
                as_name = self.__assign_as_name(full_name)
                self.__imported_names[full_name] = as_name
                self.create_import_statement(full_name, as_name, is_primitive)
            else:
                self.__imported_names[full_name] = full_name
                self.create_import_statement(full_name, '', is_primitive)


class ImportManagerPool:
    def __init__(self):
        self.__managers: Dict[str, ImportManager] = {}

    def get(self, module_name) -> ImportManager:
        if module_name in self.__managers:
            return self.__managers[module_name]
        else:
            nm = ImportManager(module_name)
            self.__managers[module_name] = nm
            return nm


class Config:
    indent: int = 4
    line_break: str = os.linesep


def indent(level: int) -> str:
    return ' ' * Config.indent * level


def indent_line(line: str, level: int) -> str:
    return f"{indent(level)}{line}" if line else ''


def indent_code(code_lines: List[str], level: int = 0) -> str:
    lines = []
    for code in code_lines:
        lines.extend(code.split(Config.line_break) if code is not None else [])
    return Config.line_break.join([indent_line(line, level) for line in lines])


def empty_lines(num: int):
    return ''.join([Config.line_break] * num)


def getter(name: str, level: int):
    lines = [
        ("@property", 0),
        (f"def {name}(self):", 0),
        (f"return self.{name}", 1),
    ]
    return indent_code([indent_line(*line) for line in lines], level)


def setter(name: str, level: int):
    lines = [
        (f"def set_{name}(self):", 0),
        (f"return self.{name}", 1),
    ]
    return indent_code([indent_line(*line) for line in lines], level)


def appender(name: str, level: int):
    lines = [
        (f"def add_{name}(self):", 0),
        (f"self.{name}.add(a_{name})", 1),
    ]
    return indent_code([indent_line(*line) for line in lines], level)


class Item:
    def __init__(self, name: str, description: Optional[str]):
        self.name: str = name
        self.description: Optional[str] = description

    def to_description(self, level: int) -> Optional[str]:
        if self.description is not None:
            return indent_code([self.description], level)
        # Returning a empty string will generate a placeholder for
        # the description.
        return ''


class Property(Item, ABC):
    def __init__(self,
                 name: str,
                 type_str: str,
                 description: Optional[str] = None,
                 default: Any = None):
        super().__init__(name, description)
        self.type_str = type_str
        self.default = default

    def to_type_str(self):
        raise NotImplementedError

    def to_access_functions(self, level):
        """ Some functions to define how to access the property values, such
        as getters, setters, len, etc.
        Args:
            level: The indentation level to format these functions.

        Returns: The access code generated for this property
        """
        name = self.name
        lines = [("@property", 0),
                 (f"def {name}(self):", 0),
                 (f"return self.{name}", 1),
                 (empty_lines(0), 0),
                 (f"@{self.name}.setter", 0),
                 (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                 (f"self.set_fields({name}={self.to_field_value()})", 1),
                 (empty_lines(0), 0)]
        return indent_code([indent_line(*line) for line in lines], level)

    def to_init_code(self, level: int) -> str:
        return indent_line(f"self.{self.name}: {self.to_type_str()} = "
                           f"{repr(self.default)}", level)

    def to_description(self, level: int) -> Optional[str]:
        desc = f"{self.name} ({self.to_type_str()})"

        if self.description is not None and self.description.strip() != '':
            desc += f"\t{self.description}"
            return indent_line(desc, level)
        return indent_line(desc, level)

    def to_field_value(self):
        raise NotImplementedError


class ClassTypeDefinition:
    def __init__(self, name: str, type_str: str,
                 description: Optional[str] = None):
        self.name = name
        self.type_str = type_str
        self.description = description

    def to_code(self, level: int) -> str:
        # type_code = f'{self.to_type_str()}'
        # type_ = f': {type_code}' if type_code.strip() != '' else ''
        return indent_line(f"{self.name} = {self.type_str}", level)

    def to_field_value(self):
        pass


class PrimitiveProperty(Property):
    TYPES = {'int', 'float', 'str', 'bool'}

    def to_type_str(self) -> str:
        return f"typing.Optional[{self.type_str}]"

    def to_field_value(self):
        if self.type_str in self.TYPES:
            return self.name
        return f"{self.name}.tid"


class CompositeProperty(Property):
    TYPES = {'List': 'typing.List'}

    def __init__(self,
                 name: str,
                 type_str: str,
                 item_type: str,
                 description: Optional[str] = None,
                 default: Any = None):
        super().__init__(name, type_str, description, default)
        self.item_type = item_type

    def to_type_str(self) -> str:
        return f"typing.Optional[{self.type_str}[{self.item_type}]]"

    def to_access_functions(self, level):
        """ Generate access function to for composite types. This extend the
        base function and add some composite specific types.
        :param level:
        :return:
        """
        base_code = super(CompositeProperty, self).to_access_functions(
            level)

        name = self.name
        lines = [
            (empty_lines(0), 0),
            (f"def num_{name}(self):", 0),
            (f"return len(self.{name})", 1),
            (empty_lines(0), 0),
            (f"def clear_{name}(self):", 0),
            (f"self.{name}.clear()", 1),
            (empty_lines(0), 0),
        ]

        if self.type_str == 'typing.List':
            lines.extend([
                (f"def add_{name}(self, a_{name}: {self.item_type}):", 0),
                (f"self.{name}.append(a_{name})", 1),
                (empty_lines(0), 0),
            ])

        add_code = indent_code([indent_line(*line) for line in lines], level)

        return base_code + add_code

    def to_field_value(self):
        item_value_str = PrimitiveProperty(
            'item', self.item_type).to_field_value()
        return f"[{item_value_str} for item in {self.name}]"


class DefinitionItem(Item):
    def __init__(self, name: str,
                 class_type: str,
                 init_args: Optional[str] = None,
                 properties: Optional[List[Property]] = None,
                 class_attributes: Optional[List[ClassTypeDefinition]] = None,
                 description: Optional[str] = None):
        super().__init__(name, description)
        self.class_type = class_type
        self.properties: List[Property] = \
            [] if properties is None else properties
        self.class_attributes = [] if class_attributes is None \
            else class_attributes
        self.description = description if description else None
        self.init_args = init_args if init_args is not None else ''
        self.init_args = self.init_args.replace('=', ' = ')

    def to_init_code(self, level: int) -> str:
        return indent_line(f"def __init__(self, {self.init_args}):", level)

    def to_code(self, level: int) -> str:
        super_args = ', '.join([item.split(':')[0].strip()
                                for item in self.init_args.split(',')])
        raw_desc = self.to_description(1)
        desc: str = '' if raw_desc is None else raw_desc
        lines = [
            empty_lines(1),
            f"__all__.extend('{self.name}')",
            empty_lines(1),
            f"class {self.name}({self.class_type}):",
        ]
        lines += [desc] if desc.strip() else []
        lines += [item.to_code(1) for item in self.class_attributes]
        lines += [empty_lines(0)]
        lines += [self.to_init_code(1),
                  indent_line(f"super().__init__({super_args})", 2)]
        lines += [item.to_init_code(2) for item in self.properties]
        lines += [empty_lines(0)]
        lines += [item.to_access_functions(1) for item in self.properties]

        return indent_code(lines, level)

    @staticmethod
    def to_item_descs(items, title):
        item_descs = [item.to_description(0) for item in items]
        item_descs = [item for item in item_descs if item is not None]
        if len(item_descs) > 0:
            item_descs = [indent_line(title, 0)] + \
                         [indent_line(desc, 1) for desc in item_descs]
        return item_descs

    def to_description(self, level: int) -> Optional[str]:
        class_desc = [] if self.description is None else [self.description]
        item_descs = self.to_item_descs(self.properties, 'Attributes:')

        # att_descs = self.to_item_descs(self.class_attributes,
        #                                'Class Attributes:')
        descs = class_desc + [empty_lines(0)] + item_descs + [empty_lines(0)]
        if len(descs) == 0:
            return ""
        quotes = indent_line('"""', 0)

        return indent_code(
            [quotes] + descs + [quotes], level)


class EntryWriter:
    """
    A writer to write entry definitions to a file.
    """

    def __init__(self,
                 package: str,
                 entry_item: DefinitionItem,
                 entry_file: str,
                 ignore_errors: Optional[List[str]],
                 description: Optional[str],
                 imports: Optional[List[str]],
                 import_manager: ImportManager
                 ):

        self.package = package
        self.description: Optional[str] = description
        self.ignore_errors: Optional[
            List[str]] = [] if not ignore_errors else ignore_errors
        self.imports: Optional[List[str]] = [] if not imports else list(
            set(imports))
        self.entry_item: DefinitionItem = entry_item
        self.entry_file_exists: bool = os.path.exists(entry_file)
        self.import_manager: ImportManager = import_manager

    def write(self, tempdir: str, destination: str, filename: str):
        """
        Write the entry information to file.

        Args:
            tempdir: A temporary directory for writing intermediate files.
            destination: The actual folder to place the generated code.
            filename: File name of the generated code.

        Returns:

        """
        # Create entry sub-directories with .generated file if the
        # subdirectory is created programmatically
        # Peak if the folder exists at the destination directory
        entry_pkg_dir = self.package.replace('.', '/')
        entry_dir_split = split_file_path(entry_pkg_dir)

        rel_dir_paths = it.accumulate(entry_dir_split, os.path.join)
        for rel_dir_path in rel_dir_paths:
            temp_path = os.path.join(tempdir, rel_dir_path)
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)

            dest_path = os.path.join(destination, rel_dir_path)
            if not os.path.exists(dest_path):
                Path(os.path.join(temp_path, '.generated')).touch()

        # Creating the file if it does not exist.
        full_path = os.path.join(tempdir, entry_pkg_dir, filename) + '.py'
        with open(full_path, 'a+') as f:
            f.write(self.to_code(0))

    def to_code(self, level: int) -> str:
        lines: List[str] = []
        if not self.entry_file_exists:
            lines = [self.to_description(0),
                     self.to_import_code(0),
                     empty_lines(1), '__all__ = []']
        lines.append(self.entry_item.to_code(0))
        return indent_code(lines, level)

    def to_description(self, level):
        quotes = '"""'
        lines = self.ignore_errors + [quotes, self.description, quotes]
        return indent_code(lines, level)

    def to_import_code(self, level):
        all_imports = set(self.import_manager.get_import_statements())
        for import_ in sorted(self.imports):
            all_imports.add(f"import {import_}")
        return indent_code(sorted(list(all_imports)), level)
