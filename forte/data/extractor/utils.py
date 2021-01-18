# Copyright 2020 The Forte Authors. All Rights Reserved.
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
"""
This file contains utility functions for extractors.
"""

from typing import Type, List, Tuple, Union, Callable
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Annotation


def bio_tagging(pack: DataPack, instance: Annotation,
    tagging_unit_type: Type[Annotation], entry_type: Type[Annotation],
    attribute: Union[Callable[[Annotation], str], str]) \
        -> List[Tuple[Annotation, str]]:
    """This utility function use BIO tagging method to convert tags
    of "instance_entry" into the same length as "instance_tagging_unit". Both
    element from "instance_entry" and "instance_tagging_unit" should Annotation
    type. This function uses their position information to
    determine B, I, O tagging for the entry on each tagging_unit element.

    Args:
        pack (Datapack): The datapack that contains the current
            instance.
        instance (Annotation): The instance from which the
            extractor will extractor feature.
        tagging_unit_type (Annotation): The type of tagging unit that entry
            tag should align to.
        entry_type (Annotation): The type of entry that contains tags.

        instance_entry (List[Annotation]): Annotations where tags come from.

    Returns:
        A list of the type List[Union[str, None], str]]. For example,
        [[None, "O"], [tag1, "B"], [tag1, "I"], [None, "O"],
         [None, "O"], [tag2, "B"], [None, "O"]]
    """
    instance_tagging_unit: List[Annotation] = \
        list(pack.get(tagging_unit_type, instance))
    instance_entry: List[Annotation] = \
        list(pack.get(entry_type, instance))

    tagged = []
    unit_id = 0
    entry_id = 0
    while unit_id < len(instance_tagging_unit) or \
        entry_id < len(instance_entry):

        if entry_id == len(instance_entry):
            tagged.append((None, 'O'))
            unit_id += 1
            continue

        is_start = True
        for unit in pack.get(tagging_unit_type, instance_entry[entry_id]):
            while instance_tagging_unit[unit_id] != unit:
                tagged.append((None, 'O'))
                unit_id += 1

            if is_start:
                location = 'B'
                is_start = False
            else:
                location = 'I'

            unit_id += 1
            if callable(attribute):
                tagged.append((attribute(instance_entry[entry_id]), location))
            else:
                tagged.append((getattr(instance_entry[entry_id], attribute), location))
        entry_id += 1
    return tagged
