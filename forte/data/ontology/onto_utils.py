from typing import List, Dict, Union, Type

from forte.data.base_pack import PackType
from forte.data.ontology.top import Entry


def record_fields(output_info: Dict[Type[Entry], Union[List, Dict]],
                  component_name: str, input_pack: PackType):
    """
    Record the fields and entries that this processor add to packs.

    Args:
        component_name: The component name that do the processing. This will be
        record here to remember which processor (or reader) alter the entries.

        output_info: The output information as specified in a reader of
        process. It contains the information of which types or fields are
        created or altered.

        input_pack: The data pack where the information will be written on.

    Returns:
    """
    for entry_type, info in output_info.items():
        fields: List[str] = []
        if isinstance(info, list):
            fields = info
        elif isinstance(info, dict):
            fields = info["fields"]
            if "component" in info.keys():
                component_name = info["component"]
        input_pack.record_fields(fields, entry_type, component_name)
