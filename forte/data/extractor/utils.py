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

from typing import List, Tuple
from ft.onto.base_ontology import Annotation, Token


def bio_tagging(pack, instance_based_on: List[Annotation],
                instance_entry: List[Annotation]) \
                -> List[Tuple[Annotation, str]]:
    """This utility function use BIO tagging method to convert tags
    of "instance_entry" into the same length as "instance_based_on". Both
    element from "instance_entry" and "instance_based_on" should Annotation
    type. This function uses their position information to
    determine B, I, O tagging for the entry on each based_on element.

    Args:
        instance_based_on (List[Annotation]): The Annotations that
            tags need to align to.

        instance_entry (List[Annotation]): Annotations where tags come from.

    Returns:
        A list of the type List[Tuple[Annotation, str]]. For example,
        [[None, "O"], [tag1, "B"], [tag1, "I"], [None, "O"],
         [None, "O"], [tag2, "B"], [None, "O"]]
    """


    tagged = []
    unit_id = 0
    entry_id = 0
    while unit_id < len(instance_based_on) or\
        entry_id < len(instance_entry):
        if entry_id == len(instance_entry):
            tagged.append((None, 'O'))
            unit_id += 1
        else:
            is_start = True
            for unit in pack.get(Token, instance_entry[entry_id]):
                while instance_based_on[unit_id] != unit:
                    tagged.append((None, 'O'))
                    unit_id += 1
                if is_start:
                    tagged.append((instance_entry[entry_id], 'B'))
                    is_start = False
                    unit_id += 1
                else:
                    tagged.append((instance_entry[entry_id], 'I'))
                    unit_id += 1


            entry_id += 1
    return tagged

    # tagged = []
    # cur_entry_id = 0
    # prev_entry_id = None
    # cur_based_on_id = 0
    # while cur_based_on_id < len(instance_based_on):
    #     base_begin = instance_based_on[cur_based_on_id].begin
    #     base_end = instance_based_on[cur_based_on_id].end

    #     if cur_entry_id < len(instance_entry):
    #         entry_begin = instance_entry[cur_entry_id].begin
    #         entry_end = instance_entry[cur_entry_id].end
    #     else:
    #         lastone = len(instance_based_on) - 1
    #         entry_begin = instance_based_on[lastone].end + 1
    #         entry_end = instance_based_on[lastone].end + 1

    #     if base_end < entry_begin:
    #         # Base: [...]
    #         # Entry       [....]
    #         tagged.append((None, "O"))
    #         prev_entry_id = None
    #         cur_based_on_id += 1
    #     elif base_begin >= entry_begin and base_end <= entry_end:
    #         # Base:    [...]
    #         # Entry:  [.......]
    #         if prev_entry_id == cur_entry_id:
    #             tagged.append((instance_entry[cur_entry_id], "I"))
    #         else:
    #             tagged.append((instance_entry[cur_entry_id], "B"))
    #         prev_entry_id = cur_entry_id
    #         cur_based_on_id += 1
    #     elif base_begin > entry_end:
    #         # Base:         [...]
    #         # Entry: [....]
    #         cur_entry_id += 1
    #     else:
    #         raise AssertionError("Unconsidered case. The entry is \
    #                     within the span of based-on entry.")
    # return tagged
