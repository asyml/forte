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

import warnings
from typing import Optional, List, Union, Tuple, Any


def bio_merge(tags: List[str], types: List[Union[str, None]],
              start: Optional[List[int]]=None, end: Optional[List[int]]=None) \
        -> Tuple[List[Any], List[Any], List[Any]]:
    r""" This function merged BIO-schemed augmented tagging scheme results and
    return entity mentions information.

    For example, BIO NER tags could be merged by passing
    tags = ['B', 'O', 'B', 'I']
    types = ['PER', '', 'LOC', 'LOC']
    start = [0, 11, 20, 24]
    end = [1, 19, 22, 26]

    After merging BIO tags, the result will be returned as
    result_types = ['PER', 'LOC']
    result_start = [0, 20]
    result_end = [1, 26]

    The function can also handle tags with no types, for example, in some word
    segmentation tasks. In this case the input `types` should be set as a list
    of None, and the returned result_type will be a list of None.


    Args:
        tags: list of bio tags, contains "B", "I", "O" labels.
        types: list of entity type, could be PER, LOC in NER task.
        start: list of start index for each input tag. default is None.
        end: list of end index for each input tag. default is None.

    Returns:
        result_types: list of merged entity type.
        result_start: list of start index for the merged entities.
        result_start: list of end index for the merged entities.
    """
    prev_type = None
    prev_tag = None
    prev_start = None
    prev_end = None
    new_entity = False
    is_indexed = True

    # No start or end information is provided, do not process index information
    if start is None or end is None:
        is_indexed = False
        start = []
        end = []
        warnings.warn('start and end indexes for the tags was not provided '
                      'and will be returned as `None`')

    # input check
    if len(tags) != len(types) or (is_indexed and (len(start) != len(tags) or
                                                   len(end) != len(tags))):
        raise ValueError('The input tags, types, start and end index have '
                         'different length, please check.')

    for tag in tags:
        if tag not in ["B", "I", "O"]:
            raise ValueError('The BIO tags contain characters beyond `BIO`, '
                             'please check the input tags.')

    result_types = []
    result_start = []
    result_end = []

    for index, (tag, type) in enumerate(zip(tags, types)):
        if tag == "B" or (tag == "I" and type != prev_type) or (tag == "O" and
                prev_tag and prev_tag != "O"):  # the last entity has ended
            if prev_tag and prev_tag != 'O':
                result_types.append(prev_type)
                result_start.append(prev_start)
                result_end.append(prev_end)

            if is_indexed:
                prev_start = start[index]
                prev_end = end[index]

            if tag != "O": # a new entity started
                new_entity = True

        elif tag == "I" and type == prev_type:  # continue with the last entity
            if is_indexed:
                prev_end = end[index]

        else:  # "O" tag
            new_entity = False

        prev_type = type
        prev_tag = tag

    if new_entity:  # check if the last entity is added in result
        result_types.append(prev_type)
        result_start.append(prev_start)
        result_end.append(prev_end)

    return result_types, result_start, result_end
