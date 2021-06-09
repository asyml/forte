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
Utility functions related to processors.
"""

__all__ = ["record_types_and_attributes_check", "collect_input_pack_record"]

from typing import Dict, Set
from forte.data.base_pack import PackType
from forte.common import ExpectedRecordNotFound
from forte.common.resources import Resources


def record_types_and_attributes_check(
    expectation: Dict[str, Set[str]], input_pack_record: Dict[str, Set[str]]
):
    r"""Check if any types or attributes in expectation dictionary doesn't
    match with input_pack.record. If not, an error of
    :class:`~forte.common.exception.ExpectedRecordNotFound` will be raised.

    Args:
        expectation: Dictionary of types and their attributes required for
            the current processor/evaluator.
        input_pack_record: The input pack record content combined with
            all the parent types and attributes collected from
            `merged_entry_tree`.

    Returns:

    """
    # pylint: disable=protected-access
    if expectation is not None:
        # check if expected types are in input pack.
        for expected_t in expectation:
            if expected_t not in input_pack_record.keys():
                raise ExpectedRecordNotFound(
                    f"The record type {expected_t} is not found in "
                    f"meta of the prediction datapack."
                )
            else:
                expected_value = expectation.get(expected_t)
                if expected_value is not None:
                    for expected_t_v in expected_value:
                        if expected_t_v not in input_pack_record.get(
                            expected_t, []
                        ):
                            raise ExpectedRecordNotFound(
                                f"The record attribute type "
                                f"{expected_t_v} is not found in "
                                f"attribute of record {expected_t} "
                                f"in meta of the input datapack."
                            )


def collect_input_pack_record(
    resources: Resources, input_pack: PackType
) -> Dict[str, Set[str]]:
    # pylint: disable=protected-access
    r"""Method to collect the type and attributes from the input pack and if
    :attr:`~forte.pipeline.Pipeline.resource` has `onto_specs` as key
    and ontology specification file path as value, then
    `merged_entry_tree` that has all the entries in ontology specification
    file would be populated. All the parent entry nodes of the input pack
    would be collected from this tree and add to the returned record
    dictionary for later comparison to enable subclass type checking.

    Args:
        resources: The pipeline attribute that stores and passes resources on
            the pipeline level.
        input_pack: The input datapack.

    Returns:
        input_pack_record: The input pack record content combined with
        all the parent types and attributes collected from
        merged_entry_tree

    """
    input_pack_record = input_pack._meta.record.copy()
    if resources.get("merged_entry_tree"):
        merged_entry_tree = resources.get("merged_entry_tree")
        merged_entry_tree.collect_parents(input_pack_record)
    return input_pack_record
