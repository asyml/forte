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
import itertools as it
import logging
import os
from abc import ABC
from pathlib import Path
from typing import Optional, Any, List, Dict, Set, Tuple

from forte.data.ontology.code_generation_exceptions import \
    CodeGenerationException
from forte.data.ontology.ontology_code_const import IGNORE_ERRORS_LINES, \
    PRIMITIVE_SUPPORTED, Config, SINGLE_COMPOSITES, COMPLEX_COMPOSITES
from forte.data.ontology.utils import split_file_path


class ImportManager:
    r"""A naive implementation that records import strings and imported names
    to be used. Mainly used to avoid import name conflicts such as:
       -- from user.module import token
       -- from system.module import token
    """

    def __init__(self, root: Optional['ImportManager'],
                 module_name: Optional[str]):
        self.__root = root
        self.__module_name = module_name
        self.__import_statements: List[str] = []
        self.__imported_names: Dict[str, str] = {}
        self.__short_name_pool: Set[str] = set()
        self.__fix_modules = False

    def fix_modules(self):
        self.__fix_modules = True

    def is_known_name(self, class_name):
        """
        Check whether the class name can be used. It will check the class name
        in both the top manager or the current manager.

        Args:
            class_name: The name to be check.

        Returns: True if the class_name can be used, which means it is either
            imported or it is of a primitive type.

        """
        return (class_name in PRIMITIVE_SUPPORTED or
                class_name in SINGLE_COMPOSITES or
                class_name in COMPLEX_COMPOSITES or
                self.is_imported(class_name))

    def is_imported(self, class_name):
        """
        Check whether this `class_name` is already imported here in the module.
        Args:
            class_name: The name to be checked for importing.

        Returns: True if the class_name is imported.

        """
        if class_name in self.__imported_names:
            return True
        elif self.__root is None:
            return False
        else:
            if self.__root.is_imported(class_name) and not self.__fix_modules:
                self.add_object_to_import(class_name)
                return True

    def all_stored(self):
        return self.__imported_names.items()

    def get_name_to_use(self, full_name):
        if full_name in PRIMITIVE_SUPPORTED:
            return full_name

        if self.is_imported(full_name):
            return self.__imported_names[full_name]

    def get_import_statements(self):
        return sorted(self.__import_statements)

    def create_import_statement(self, full_name: str, as_name: str):
        if full_name not in PRIMITIVE_SUPPORTED:
            parts = full_name.split('.')
            class_name = parts[-1]

            if len(parts) > 1:
                module_name = '.'.join(parts[:-1])

                if (self.__module_name is None or
                        not module_name == self.__module_name):
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

    def add_object_to_import(self, full_name: str):
        if self.__fix_modules:
            # After fix the module, we should not add objects for import.
            raise CodeGenerationException(
                f'The module [{self.__module_name}] is fixed, cannot add '
                f'more objects.')

        if full_name not in self.__imported_names:
            if full_name not in PRIMITIVE_SUPPORTED:
                as_name = self.__assign_as_name(full_name)
                self.__imported_names[full_name] = as_name
                self.create_import_statement(full_name, as_name)


class ImportManagerPool:
    """
    Holds import managers. The top_manager stores the default imports
    guaranteed by Forte's system ontology. The additional import managers are
    further populated by analyzing the generated code.
    """

    def __init__(self):
        self.__root_manager = ImportManager(None, None)
        self.__managers: Dict[str, ImportManager] = {}

    @property
    def root(self) -> ImportManager:
        return self.__root_manager

    def get(self, module_name: str) -> ImportManager:
        if module_name in self.__managers:
            return self.__managers[module_name]
        else:
            nm = ImportManager(self.__root_manager, module_name)
            self.__managers[module_name] = nm
            return nm

    def fix_all_modules(self):
        self.__root_manager.fix_modules()
        for im in self.__managers.values():
            im.fix_modules()


def indent(level: int) -> str:
    return ' ' * Config.indent * level


def indent_line(line: str, level: int) -> str:
    return f"{indent(level)}{line}" if line else ''


def indent_code(code_lines: List[str], level: int = 0, ending='\n') -> str:
    lines = []

    for code in code_lines:
        if code == '':
            lines.append('')
        else:
            lines.extend(
                code.split(Config.line_break) if code is not None else []
            )

    return Config.line_break.join(
        [indent_line(line, level) for line in lines]) + ending


def getter(name, field_name):
    # Construct getter.
    return [
        ("@property", 0),
        (f"def {name}(self):", 0),
        (f"return self.{field_name}", 1),
        ('', 0),
    ]


def change_get_state(name, field_name, level):
    return [
        (f"state['{name}'] = self.{field_name}", level)
    ]


def change_set_state(name, field_name, level):
    return [
        (f"self.{field_name} = state.get('{name}', None) ", level)
    ]


class EntryName:
    def __init__(self, entry_name: str):
        entry_splits = entry_name.split('.')
        self.filename, self.name = entry_splits[-2:]
        self.pkg = '.'.join(entry_splits[0: -2])
        self.pkg_dir = self.pkg.replace('.', '/')
        self.module_name: str = f"{self.pkg}.{self.filename}"
        self.class_name: str = entry_name


class Item:
    def __init__(self, name: str, description: Optional[str]):
        self.name: str = name
        self.description: Optional[str] = description

    @property
    def field_name(self):
        return self.name

    def to_description(self, level: int) -> Optional[str]:
        if self.description is not None:
            return indent_code([self.description], level)
        # Returning a empty string will generate a placeholder for
        # the description.
        return ''


class Property(Item, ABC):
    def __init__(self, import_manager: ImportManager,
                 name: str, type_str: str, description: Optional[str] = None,
                 default_val: Any = None):
        super().__init__(name, description)
        self.type_str = type_str
        self.default_val = default_val
        self.import_manager: ImportManager = import_manager

    def to_type_str(self):
        raise NotImplementedError

    def to_access_functions(self, level):
        """ Some functions to define how to access the property values, such
        as getters, setters, len, etc.
        Args:
            level: The indentation level to format these functions.

        Returns: The access code generated for this property
        """
        raise NotImplementedError

    def to_getstate(self, level):
        # return change_get_state(self.name, self.field_name, level)
        return [
            (f"state['{self.name}'] = self.{self.field_name}", level)
        ]

    def to_setstate(self, level):
        # return change_set_state(self.name, self.field_name, level)
        return [
            (f"self.{self.field_name} = state.get('{self.name}', None) ", level)
        ]

    def to_init_code(self, level: int) -> str:
        return indent_line(f"self.{self.field_name}: {self.to_type_str()} = "
                           f"{repr(self.default_val)}", level)

    def to_description(self, level: int) -> Optional[str]:
        desc = f"{self.field_name} ({self.to_type_str()})"

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
        return indent_code([f"{self.name} = {self.type_str}"], level)

    def to_field_value(self):
        pass


class NonCompositeProperty(Property):
    def __init__(self, import_manager: ImportManager,
                 name: str, type_str: str, description: Optional[str] = None,
                 default_val: Any = None):
        super(NonCompositeProperty, self).__init__(
            import_manager, name, type_str, description, default_val)

        # Primitive type will use optional in type string, so we add the
        # optional here.
        self.option_type = 'typing.Optional'
        import_manager.add_object_to_import('typing.Optional')

        self.is_forte_type = import_manager.is_imported(type_str)

    def to_type_str(self) -> str:
        option_type = self.import_manager.get_name_to_use(self.option_type)
        if self.is_forte_type:
            return f"{option_type}[int]"
        else:
            return f"{option_type}[{self.type_str}]"

    def to_access_functions(self, level):
        name = self.name

        if self.is_forte_type:
            type_to_use = self.import_manager.get_name_to_use(self.type_str)
            lines = [
                ("@property", 0),
                (f"def {name}(self):", 0),
                (f"return self.__pack.get_entry(self.{name})", 1),
                ('', 0),
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {type_to_use}):", 0),
                (f"self.set_fields({self.field_name}="
                 f"self.__pack.add_entry({name}))", 1),
            ]
        else:
            lines = getter(name, self.field_name)
            lines.extend([
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                (f"self.set_fields({self.field_name}"
                 f"={self.to_field_value()})", 1),
            ])
        return indent_code([indent_line(*line) for line in lines], level)

    def to_field_value(self):
        return self.name


class DictProperty(Property):
    def __init__(self,
                 import_manager: ImportManager,
                 name: str,
                 key_type: str,
                 value_type: str,
                 description: Optional[str] = None,
                 default_val: Any = None,
                 self_ref: bool = False):
        super().__init__(import_manager, name, 'typing.Dict',
                         description=description,
                         default_val=default_val)
        self.key_type: str = key_type
        self.value_type: str = value_type
        self.self_ref: bool = self_ref

        self.value_is_forte_type = import_manager.is_imported(self.value_type)

    def to_type_str(self) -> str:
        option_type = self.import_manager.get_name_to_use('typing.Optional')
        composite_type = self.import_manager.get_name_to_use(self.type_str)

        key_type = self.import_manager.get_name_to_use(self.key_type)
        value_type = self.import_manager.get_name_to_use(self.value_type)

        if self.self_ref:
            value_type = '"' + value_type + '"'

        if self.value_is_forte_type:
            return f"{option_type}[{composite_type}[{key_type}, int]]"
        else:
            return f"{option_type}[{composite_type}[{key_type}, {value_type}]]"

    def to_field_value(self):
        return self.name

    def to_access_functions(self, level):
        """ Generate access function to for Dict types. This extend the
        base function and add some composite specific types.

        :param level:
        :return:
        """
        name = self.name
        key_type = self.import_manager.get_name_to_use(self.key_type)
        value_type = self.import_manager.get_name_to_use(self.value_type)

        if self.self_ref:
            value_type = '"' + value_type + '"'

        lines = getter(name, self.field_name)

        # Construct setter.
        if self.value_is_forte_type:
            lines.extend([
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                (f"self.set_fields("
                 f"{self.field_name}="
                 f"dict([(k, self.__pack.add_entry_(v)) "
                 f"for k, v in {name}.items()]))", 1),
                ('', 0),
            ])
        else:
            lines.extend([
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                (f"self.set_fields("
                 f"{self.field_name}={name})", 1),
                ('', 0),
            ])

        # Construct counter.
        lines.extend([
            (f"def num_{name}(self):", 0),
            (f"return len(self.{self.field_name})", 1),
            ('', 0),
        ])

        # Construct clear (deletion).
        if self.value_is_forte_type:
            lines.extend([
                (f"def clear_{name}(self):", 0),
                (f"[self.__pack.delete_entry("
                 f"self.__pack.get_entry(tid)) for tid in self.{name}.values()]",
                 1),
                (f"self.{self.field_name}.clear()", 1),
            ])
        else:
            lines.extend([
                (f"def clear_{name}(self):", 0),
                (f"self.{self.field_name}.clear()", 1),
            ])

        # Construct appender.
        if self.value_is_forte_type:
            lines.extend([
                ('', 0),
                (f"def add_{name}(self, key: {key_type}, value: {value_type}):",
                 0),
                (f"self.{self.field_name}[key] = self.__pack.add_entry_(value)",
                 1),
            ])
        else:
            lines.extend([
                ('', 0),
                (f"def add_{name}(self, key: {key_type}, value: {value_type}):",
                 0),
                (f"self.{name}[key] = value", 1),
            ])

        return indent_code([indent_line(*line) for line in lines], level)


class ListProperty(Property):
    def __init__(self,
                 import_manager: ImportManager,
                 name: str,
                 type_str: str,
                 item_type: str,
                 description: Optional[str] = None,
                 default_val: Any = None,
                 self_ref: bool = False):
        super().__init__(import_manager, name, type_str,
                         description=description,
                         default_val=default_val)
        self.item_type: str = item_type
        self.is_forte_type = import_manager.is_imported(self.item_type)

        # self_ref would probably not happen, because we are using int for
        # entry types.
        self.self_ref: bool = self_ref

    def to_type_str(self) -> str:
        option_type = self.import_manager.get_name_to_use('typing.Optional')
        composite_type = self.import_manager.get_name_to_use(self.type_str)
        item_type = self.import_manager.get_name_to_use(self.item_type)

        if self.self_ref:
            item_type = '"' + item_type + '"'

        if self.is_forte_type:
            return f"{option_type}[{composite_type}[int]]"
        else:
            return f"{option_type}[{composite_type}[{item_type}]]"

    def to_access_functions(self, level):
        """ Generate access function to for composite types. This extend the
        base function and add some composite specific types.
        :param level:
        :return:
        """
        name = self.name
        lines = getter(name, self.field_name)

        # Construct setter.
        if self.is_forte_type:
            lines.extend([
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                (f"self.set_fields("
                 f"{self.field_name}="
                 f"[self.__pack.add_entry_(obj) for obj in {name}])", 1),
                ('', 0),
            ])
        else:
            lines.extend([
                (f"@{self.name}.setter", 0),
                (f"def {name}(self, {name}: {self.to_type_str()}):", 0),
                (f"self.set_fields("
                 f"{self.field_name}={name})", 1),
                ('', 0),
            ])

        # Construct counter.
        lines.extend([
            (f"def num_{name}(self):", 0),
            (f"return len(self.{self.field_name})", 1),
            ('', 0),
        ])

        # Construct clear (deletion).
        if self.is_forte_type:
            lines.extend([
                (f"def clear_{name}(self):", 0),
                (f"[self.__pack.delete_entry("
                 f"self.__pack.get_entry(tid)) "
                 f"for tid in self.{self.field_name}]", 1),
                (f"self.{self.field_name}.clear()", 1),
            ])
        else:
            lines.extend([
                (f"def clear_{name}(self):", 0),
                (f"self.{self.field_name}.clear()", 1),
            ])

        # Construct appender.
        item_type = self.import_manager.get_name_to_use(self.item_type)
        if self.self_ref:
            item_type = '"' + item_type + '"'

        if self.is_forte_type:
            lines.extend([
                ('', 0),
                (f"def add_{name}(self, a_{name}: {item_type}):", 0),
                (f"self.{self.field_name}.append("
                 f"self.__pack.add_entry_(a_{name}))", 1),
            ])
        else:
            lines.extend([
                ('', 0),
                (f"def add_{name}(self, a_{name}: {item_type}):", 0),
                (f"self.{self.field_name}.append(a_{name})", 1),
            ])

        return indent_code([indent_line(*line) for line in lines], level)

    def to_field_value(self):
        # item_value_str = PrimitiveProperty(self.import_manager, 'item',
        #                                    self.item_type).to_field_value()
        # print('field vale')
        # print(item_value_str)
        # print(f"[{item_value_str} for item in {self.name}]")
        # return f"[{item_value_str} for item in {self.name}]"
        return self.name


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

    def to_get_state_code(self, level: int) -> str:
        lines = [
            ("def __getstate__(self): ", 0),
            ("state = super().__getstate__()", 1),
        ]

        for p in self.properties:
            lines.extend(p.to_getstate(1))
        lines.append(("return state", 1))
        return indent_code([indent_line(*line) for line in lines], level)

    def to_set_state_code(self, level: int) -> str:
        lines = [
            ("def __setstate__(self, state): ", 0),
            ("state = super().__setstate__(state)", 1),
        ]

        for p in self.properties:
            lines.extend(p.to_setstate(1))

        return indent_code([indent_line(*line) for line in lines], level)

    def to_code(self, level: int) -> str:
        super_args = ', '.join([item.split(':')[0].strip()
                                for item in self.init_args.split(',')])
        raw_desc = self.to_description(1)
        desc: str = '' if raw_desc is None else raw_desc
        lines = [
            '',
            '',
            f"class {self.name}({self.class_type}):",
        ]
        lines += [desc] if desc.strip() else []
        lines += [item.to_code(1) for item in self.class_attributes]
        lines += [self.to_init_code(1),
                  indent_line(f"super().__init__({super_args})", 2)]
        lines += [item.to_init_code(2) for item in self.properties]
        lines += ['']
        lines += [self.to_get_state_code(1)]
        lines += [self.to_set_state_code(1)]
        lines += [item.to_access_functions(1) for item in self.properties]
        return indent_code(lines, level, '')

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

        descs = class_desc + [''] + item_descs + ['']
        if len(descs) == 0:
            return ""
        quotes = indent_line('"""', 0)

        return indent_code(
            [quotes] + descs + [quotes], level)


class ModuleWriter:
    """
    A writer to write entry definitions to a file.
    """

    def __init__(self, module_name: str,
                 import_managers: ImportManagerPool):
        self.module_name = module_name
        self.description: Optional[str] = None
        self.import_managers: ImportManagerPool = import_managers
        self.entries: List[Tuple[EntryName, DefinitionItem]] = []

        pkg, self.file_name = self.module_name.rsplit('.', 1)
        self.pkg_dir = pkg.replace('.', '/')

    def set_description(self, description: str):
        self.description = description

    def add_entry(self, entry_name: EntryName, entry_item: DefinitionItem):
        self.entries.append((entry_name, entry_item))

    def make_module_dirs(self, tempdir: str, destination: str):
        """
        Create entry sub-directories with .generated file to indicate the
         subdirectory is created by this procedure. No such file will be added
         if the directory already exists.

        Args:
            tempdir: A temp directory to create the structure, code will be
              first generated here.
            destination: The destination directory where the code should be
              placed

        Returns:

        """

        entry_dir_split = split_file_path(self.pkg_dir)

        rel_dir_paths = it.accumulate(entry_dir_split, os.path.join)
        for rel_dir_path in rel_dir_paths:
            temp_path = os.path.join(tempdir, rel_dir_path)
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)

            dest_path = os.path.join(destination, rel_dir_path)
            if not os.path.exists(dest_path):
                Path(os.path.join(temp_path, '.generated')).touch()

    def write(self, tempdir: str, destination: str):
        """
        Write the entry information to file.

        Args:
            tempdir: A temporary directory for writing intermediate files.
            destination: The actual folder to place the generated code.

        Returns:

        """

        self.make_module_dirs(tempdir, destination)
        full_path = os.path.join(tempdir, self.pkg_dir, self.file_name) + '.py'

        with open(full_path, 'w') as f:
            # Write header.
            f.write(self.to_header(0))
            for entry_name, entry_item in self.entries:
                logging.info('Writing class: ' + entry_name.class_name)
                f.write(entry_item.to_code(0))

    def to_header(self, level: int) -> str:
        all_first_line = indent_line('__all__ = [', 0)
        all_mid_lines = indent_code(
            [f'"{en.name}",' for en, ei in self.entries], 1, '')
        all_last_line = indent_line(']', 0)

        lines = [self.to_description(0),
                 self.to_import_code(0),
                 '',
                 all_first_line,
                 all_mid_lines,
                 all_last_line,
                 ]
        return indent_code(lines, level)

    def to_description(self, level):
        quotes = '"""'
        lines = IGNORE_ERRORS_LINES + [quotes, self.description, quotes]
        return indent_code(lines, level)

    def to_import_code(self, level):
        return indent_code(
            self.import_managers.get(self.module_name).get_import_statements(),
            level)


class ModuleWriterPool:
    def __init__(self, import_managers: ImportManagerPool):
        self.__module_writers: Dict[str, ModuleWriter] = {}
        self.__import_managers = import_managers

    def get(self, module_name: str) -> ModuleWriter:
        if module_name in self.__module_writers:
            return self.__module_writers[module_name]
        else:
            mw = ModuleWriter(
                module_name, self.__import_managers)
            self.__module_writers[module_name] = mw
            return mw

    def writers(self):
        return self.__module_writers.values()
