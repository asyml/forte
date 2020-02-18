import os
from typing import List, Set

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import top

REQUIRED_IMPORTS: List[str] = [
    'typing',
    # 'ft.onto',
    # 'forte.data.data_pack',
]

TOP_MOST_CLASS = 'forte.data.ontology.core.Entry'
TOP_MOST_MODULE_NAME = 'forte.data.ontology.core'

DEFAULT_CONSTRAINTS_KEYS = {
    "BaseLink": {"parent_type": "ParentType", "child_type": "ChildType"},
    "BaseGroup": {"member_type": "MemberType"}
}
AUTO_GEN_SIGNATURE = '***automatically_generated***'
IGNORE_ERRORS_LINES: List[str] = [
    f'# {AUTO_GEN_SIGNATURE}',
    '# flake8: noqa',
    '# mypy: ignore-errors',
    '# pylint: skip-file']
DEFAULT_PREFIX = "ft.onto"

PRIMITIVE_SUPPORTED: Set[str] = {'int', 'float', 'str', 'bool'}

SINGLE_COMPOSITES = {'List': 'typing.List'}

COMPLEX_COMPOSITES = {'Dict': 'typing.Dict'}


class SchemaKeywords:
    ontology_name = 'ontology_name'
    prefixes = 'additional_prefixes'
    definitions = 'definitions'
    parent_entry = 'parent_entry'
    entry_name = 'entry_name'
    description = 'description'
    attributes = 'attributes'
    attribute_name = 'name'
    attribute_type = 'type'
    parent_type = 'parent_type'
    child_type = 'child_type'
    default_value = 'default'
    element_type = 'item_type'
    dict_key_type = 'key_type'
    dict_value_type = 'value_type'


def file_header(ontology_description, ontology_name):
    return (
        f'{ontology_description}\n\n'
        f'Automatically generated ontology {ontology_name}. '
        f'Do not change manually.'
    )


def class_name(clazz):
    return '.'.join((clazz.__module__, clazz.__name__))


SINGLE_PACK_CLASSES = [class_name(clazz) for clazz in top.SinglePackEntries]
MULTI_PACK_CLASSES = [class_name(clazz) for clazz in top.MultiPackEntries]

PACK_TYPE_CLASS_NAME = class_name(PackType)

def hardcoded_pack_map(clazz):
    if clazz in SINGLE_PACK_CLASSES:
        return class_name(DataPack)
    elif clazz in MULTI_PACK_CLASSES:
        return class_name(MultiPack)
    else:
        # When not found, return the default.
        return PACK_TYPE_CLASS_NAME


class Config:
    indent: int = 4
    line_break: str = os.linesep
