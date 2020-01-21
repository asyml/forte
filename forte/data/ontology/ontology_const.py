from typing import List

REQUIRED_IMPORTS: List[str] = [
    'typing',
    # 'ft.onto',
    # 'forte.data.data_pack',
]
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
