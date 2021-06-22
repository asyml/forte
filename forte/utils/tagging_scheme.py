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

import logging
from typing import Optional, List, Union, Tuple


def bio_merge(
    tags: List[str],
    types: List[Union[str, None]],
    index: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[
    List[Union[str, None]], List[Tuple[Union[int, None], Union[int, None]]]
]:
    r"""This function merged BIO-schemed augmented tagging scheme results and
    return chunks information.

    For example, BIO NER tags could be merged by passing
    tags = ['B', 'O', 'B', 'I']
    types = ['PER', '', 'LOC', 'LOC']
    index = [(0, 1), (11, 19), (20, 22), (24, 26)]

    After merging BIO tags, the results will be returned as
    result_types = ['PER', 'LOC']
    result_index = [(0, 1), (20, 26)]

    The function handles 'I' with no leading 'B' tag. If we encounter
    "I" while its type is different from the previous type, we will consider
    this "I" as a "B" and start a new record here.

    The function can also handle tags with no types, for example, in some word
    segmentation tasks. In this case the input `types` should be set as a list
    of None, and the returned result_type will be a list of None.

    Args:
        tags: list of bio tags, contains "B", "I", "O" labels.
        types: list of entity type, could be PER, LOC in NER task.
        index: list of (start, end) index for each input tag. default is None.

    Returns:
        result_types: list of merged entity type.
        result_index: list of (start, end) index for the merged entities.
    """
    prev_type: Optional[str] = None
    prev_tag: Optional[str] = None
    prev_start: Optional[int] = None
    prev_end: Optional[int] = None
    new_entity: bool = False
    is_indexed: bool = True

    # No start or end information is provided, do not process index information
    if index is None:
        is_indexed = False
        start = []
        end = []
        logging.warning(
            "start and end indexes for the tags was not provided "
            "and will be returned as `None`"
        )
    else:  # get start and end index
        start, end = zip(*index)

    # input check
    if len(tags) != len(types) or (
        is_indexed and (len(start) != len(tags) or len(end) != len(tags))
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

    result_types: List[Union[str, None]] = []
    result_start: List[Union[int, None]] = []
    result_end: List[Union[int, None]] = []

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

            if is_indexed:
                prev_start = start[idx]
                prev_end = end[idx]

            if tag != "O":  # a new entity started
                new_entity = True

        elif tag == "I" and type == prev_type:  # continue with the last entity
            if is_indexed:
                prev_end = end[idx]

        else:  # "O" tag
            new_entity = False

        prev_type = type
        prev_tag = tag

    if new_entity:  # check if the last entity is added in result
        result_types.append(prev_type)
        result_start.append(prev_start)
        result_end.append(prev_end)

    result_index: List[Tuple[Union[int, None], Union[int, None]]] = list(
        zip(result_start, result_end)
    )

    return result_types, result_index
