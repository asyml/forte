# Copyright 2021 The Forte Authors. All Rights Reserved.
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
Utility functions related to tagging scheme.
"""

from typing import Optional, List, Tuple


def bio_merge(
    tags: List[str],
    types: List[str],
    indices: List[Tuple[int, int]],
) -> Tuple[List[Optional[str]], List[Tuple[int, int]]]:
    r"""This function merged BIO-schemed augmented tagging scheme results and
    return chunks information.

    For example, BIO NER tags could be merged by passing
    tags = ['B', 'O', 'B', 'I']
    types = ['PER', '', 'LOC', 'LOC']
    indices = [(0, 1), (11, 19), (20, 22), (24, 26)]

    After merging BIO tags, the results will be returned as
    result_types = ['PER', 'LOC']
    result_indices = [(0, 1), (20, 26)]

    The function handles 'I' with no leading 'B' tag. If we encounter
    "I" while its type is different from the previous type, we will consider
    this "I" as a "B" and start a new record here.

    The function can also handle tags with no types, for example, in some word
    segmentation tasks. In this case the input `types` should be set as a list
    of empty string "", and the returned `result_types` will be an empty list.

    Args:
        tags: list of bio tags, contains "B", "I", "O" labels.
        types: list of entity type, could be PER, LOC in NER task.
        indices: list of (start, end) index for each input tag.

    Returns:
        result_types: list of merged entity type.
        result_indices: list of (start, end) index for the merged entities.
    """
    prev_type: Optional[str] = None
    prev_tag: Optional[str] = None
    prev_start: int = -1
    prev_end: int = -1
    new_entity: bool = False
    start, end = zip(*indices)

    # input check
    if len(tags) != len(types) or (
        (len(start) != len(tags) or len(end) != len(tags))
    ):
        raise ValueError(
            "The input tags, types, start and end index have "
            "different length, please check."
        )

    for tag in tags:
        if tag not in ["B", "I", "O"]:
            raise ValueError(
                "The BIO tags contain characters beyond `BIO`, "
                "please check the input tags."
            )

    result_types: List[Optional[str]] = []
    result_start: List[int] = []
    result_end: List[int] = []

    for idx, (tag, type) in enumerate(zip(tags, types)):
        if (
            tag == "B"
            or (tag == "I" and type != prev_type)
            or (tag == "O" and prev_tag and prev_tag != "O")
        ):  # the last entity has ended
            if prev_tag and prev_tag != "O":
                result_types.append(prev_type)
                result_start.append(prev_start)
                result_end.append(prev_end)

            new_entity = tag != "O"  # a new entity started

            prev_start = start[idx]
            prev_end = end[idx]

        elif tag == "I" and type == prev_type:  # continue with the last entity
            prev_end = end[idx]
            if prev_tag == "O":  # edge case: no leading B, start a new entity
                new_entity = True
                prev_start = start[idx]

        else:  # "O" tag
            new_entity = False

        prev_type = type
        prev_tag = tag

    if new_entity:  # check if the last entity is added in result
        result_types.append(prev_type)
        result_start.append(prev_start)
        result_end.append(prev_end)

    result_indices: List[Tuple[int, int]] = list(zip(result_start, result_end))

    # no types provided, return empty list
    if len(set(result_types)) == 1 and result_types[0] == "":
        result_types = []

    return result_types, result_indices
