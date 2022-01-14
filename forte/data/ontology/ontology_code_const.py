import os
from typing import List, Set

from string import Template

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import top
from forte.data.ontology import utils


class SchemaKeywords:
    ontology_name = "name"
    imports = "imports"
    prefixes = "additional_prefixes"
    definitions = "definitions"
    parent_entry = "parent_entry"
    entry_name = "entry_name"
    description = "description"
    attributes = "attributes"
    attribute_name = "name"
    attribute_type = "type"
    parent_type = "parent_type"
    child_type = "child_type"
    member_type = "member_type"
    default_value = "default"
    element_type = "item_type"
    dict_key_type = "key_type"
    dict_value_type = "value_type"
    ndarray_dtype = "ndarray_dtype"
    ndarray_shape = "ndarray_shape"


# Some names are used as properties by the core types, they should not be
# reused by attributes.
RESERVED_ATTRIBUTE_NAMES: Set[str] = {
    "text",
    "span",
    "begin",
    "end",
    "index_key",
    "parent",
    "child",
    "embedding",
    "tid",
    "pack",
    "pack_id",
    "index_key",
}

REQUIRED_IMPORTS: List[str] = ["typing"]

TOP_MOST_MODULE_NAME = "forte.data.ontology.core"

DEFAULT_CONSTRAINTS_KEYS = {
    "BaseLink": {
        SchemaKeywords.parent_type: "ParentType",
        SchemaKeywords.child_type: "ChildType",
    },
    "BaseGroup": {SchemaKeywords.member_type: "MemberType"},
}

AUTO_GEN_SIGNATURE = "***automatically_generated***"
AUTO_GEN_FILENAME = ".generated"
AUTO_DEL_FILENAME = ".deleted"

SOURCE_JSON_PFX = "***source json:"
SOURCE_JSON_SFX = "***"
SOURCE_JSON_TEMP = Template(f"{SOURCE_JSON_PFX}$file_path{SOURCE_JSON_SFX}")


def get_ignore_error_lines(json_filepath: str) -> List[str]:
    source_json_sign = SOURCE_JSON_TEMP.substitute(file_path=json_filepath)
    return [
        f"# {AUTO_GEN_SIGNATURE}",
        f"# {source_json_sign}",
        "# flake8: noqa",
        "# mypy: ignore-errors",
        "# pylint: skip-file",
    ]


DEFAULT_PREFIX = "ft.onto"

SUPPORTED_PRIMITIVES = {"int", "float", "str", "bool"}
NON_COMPOSITES = {key: key for key in SUPPORTED_PRIMITIVES}
COMPOSITES = {"List", "Dict", "NdArray"}

ALL_INBUILT_TYPES = set(list(NON_COMPOSITES.keys()) + list(COMPOSITES))


def file_header(desc_str, ontology_name):
    desc_str = "" if desc_str is None else desc_str.strip()
    desc_str = desc_str + "\n" if desc_str else ""
    return (
        f"{desc_str}"
        f"Automatically generated ontology {ontology_name}. "
        f"Do not change manually."
    )


def class_name(clazz):
    return ".".join((clazz.__module__, clazz.__name__))


SINGLE_PACK_CLASSES = [class_name(clazz) for clazz in top.SinglePackEntries]
MULTI_PACK_CLASSES = [class_name(clazz) for clazz in top.MultiPackEntries]

major_version, minor_version = utils.get_python_version()
if major_version >= 3 and minor_version >= 7:
    PACK_TYPE_CLASS_NAME = class_name(PackType)
else:
    # bug in python < 3.7
    # returns    => typing.TypeVar('').__module__ == 'typing' (wrong)
    # instead of => typing.TypeVar('').__module__ == 'forte.data.base_pack'
    PACK_TYPE_CLASS_NAME = "forte.data.base_pack.PackType"


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
