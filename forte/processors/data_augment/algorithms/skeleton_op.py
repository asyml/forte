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
SkeletonOp is the most low level Op used for Data Augmentation.
Through this Op, users can use in-built utility functions to implement
their own augmentation logic.
"""

from collections import defaultdict
from typing import DefaultDict, Tuple, Union, Dict, Any, Set
from abc import abstractmethod, ABC
from forte.data.data_pack import DataPack
from forte.data.ontology.top import (
    Annotation,
)
from forte.common.configuration import Config
from forte.data.span import Span
from bisect import bisect_left

from sortedcontainers.sorteddict import SortedDict
from sortedcontainers.sortedlist import SortedList

__all__ = ["SkeletonOp"]


class SkeletonOp(ABC):
    r"""
    The SkeletonOp is the most basic augmentation Op that
    gives users the most amount of freedom to implement their
    logic of augmentation. The users are expected to use the
    provided utility functions to implement their own augmentation
    logic which will then ne substantiated into new data packs. This
    Op requires the users to have a relatively stronger understanding of
    Forte's internal setup.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:

        # :attr:`_replaced_annos`: {datapack id: SortedList[span, new text]}
        # It records the spans replaced by new texts.
        # It is a map from datapack id to a list of tuples
        # (span, new text) inserted by :func:`replace`.
        # The new text will be used for building new data pack.
        self._replaced_annos: DefaultDict[
            int, SortedList[Tuple[Span, str]]
        ] = defaultdict(lambda: SortedList([], key=lambda x: x[0]))

        # :attr:`_inserted_annos_pos_len`: {datapack id: Dict{position: length}}
        # It records the position and length of inserted spans,
        # mapping from datapack id to a dictionary (position -> length)
        # inserted by :func:`insert`.
        # The position is the index in the original datapack
        # of insertion, and the length is the length of the inserted string.
        self._inserted_annos_pos_len: DefaultDict[
            int, SortedDict[int, int]
        ] = defaultdict(lambda: SortedDict({}))

        # :attr:`_deleted_annos_id`: {datapack id: Set[annotation tid]}
        # It records the deleted span ids, mapping from datapack id
        # to a set of annotation tids appended by :func:`delete`.
        self._deleted_annos_id: DefaultDict[int, Set[int]] = defaultdict(set)

        # :attr:`_data_pack_map`: {orig pack id: new pack id}
        # It maintains a mapping from the pack id
        # of the original pack to the pack id of augmented pack.
        # It is used when copying the MultiPackLink and MultiPackGroup.
        self._data_pack_map: Dict[int, int] = {}

        # :attr:`_entry_maps`: {datapack id: Dict{orig tid, new tid}}
        # It is a map for tracking the annotation ids
        # before and after the auto align. It maps the
        # original annotation tid to the new annotation tid.
        # It is used when copying the Link/Group/MultiPackLink/MultiPackGroup.
        self._entry_maps: Dict[int, Dict[int, int]] = {}

        self.configs = configs

    def _overlap_with_existing(self, pid: int, begin: int, end: int) -> bool:
        r"""
        This function will check whether the new span
        has an overlap with any existing spans.
        Args:
            pid: Datapack Id.
            begin: The span begin index.
            end: The span end index.
        Returns:
            True if the input span overlaps with
            any existing spans, False otherwise.
        """
        if len(self._replaced_annos[pid]) == 0:
            return False
        ind: int = (
            bisect_left(self._replaced_annos[pid], (Span(begin, begin), "")) - 1
        )
        if ind < 0:
            ind += 1

        while ind < len(self._replaced_annos[pid]):
            span: Span = self._replaced_annos[pid][ind][0]
            if not (span.begin >= end or span.end <= begin):
                return True
            if span.begin > end:
                break
            ind += 1

        return False

    def inserted_annotation_status(self):
        r"""
        This function is used to return the current
        status of the sorted dict that holds information
        regarding the inserted annotations.
        Returns:
            DefaultDict[int, SortedDict[int, int]]
        """
        return self._inserted_annos_pos_len

    def replaced_annotation_status(self):
        r"""
        This function is used to return the current
        status of the sorted dict that holds information
        regarding the replaced annotations.
        Returns:
            DefaultDict[int, SortedList[Tuple[Span, str]]]
        """
        return self._replaced_annos

    def deleted_annotation_status(self):
        r"""
        This function is used to return the current
        status of the sorted dict that holds information
        regarding the deleted annotations.
        Returns:
            DefaultDict[DefaultDict[int, Set[int]]]
        """
        return self._deleted_annos_id

    def insert_annotations(
        self, inserted_text: str, data_pack: DataPack, pos: int
    ) -> bool:
        r"""
        This is a utility function to insert a new annotation. After
        getting the inserted text, it will register the input & output
        for later batch process of building the new data pack.
        The insertion at each position can only occur once. If there
        is already an insertion at current position, it will abort the
        insertion and return False.
        Args:
            inserted_text: The text string to insert.
            data_pack: The datapack for insertion.
            pos: The position(index) of insertion.
        Returns:
            A bool value. True if the insertion happened, False otherwise.
        """

        pid: int = data_pack.pack_id
        if self._overlap_with_existing(pid, pos, pos):
            return False

        if pos not in self._inserted_annos_pos_len[pid]:
            self._replaced_annos[pid].add((Span(pos, pos), inserted_text))
            self._inserted_annos_pos_len[pid][pos] = len(inserted_text)
            return True
        return False

    def delete_annotations(self, input_anno: Annotation) -> bool:
        r"""
        This is a utility function to delete an annotation. If the same
        annotation is tried to be deleted twice, the function will terminate
        and return False.
        Args:
            input_anno: The annotation to remove.
        Returns:
            A bool value. True if the deletion happened, False otherwise.
        """
        pid: int = input_anno.pack.pack_id
        if self._overlap_with_existing(pid, input_anno.begin, input_anno.end):
            return False

        self._replaced_annos[pid].add((input_anno.span, ""))
        self._deleted_annos_id[pid].add(input_anno.tid)
        return True

    def replace_annotations(
        self, replacement_anno: Annotation, replaced_text
    ) -> bool:
        r"""
        This is a utility function to record specifically a replacement of an annotation.
        With this function, an annotation can be replaced with another annotation.
        If the same annotation is tried to be replaced twice, the function will terminate
        and return False.
        Args:
            input_anno: The annotation to replace.
        Returns:
            A bool value. True if the replacement happened, False otherwise.
        """
        pid: int = replacement_anno.pack.pack_id
        if self._overlap_with_existing(
            pid, replacement_anno.begin, replacement_anno.end
        ):
            return False

        self._replaced_annos[pid].add(
            (Span(replacement_anno.begin, replacement_anno.end), replaced_text)
        )
        return True

    @abstractmethod
    def augment(self, data_pack: DataPack) -> bool:
        r"""
        This method is left to be implemented by the user of this Op.
        The user can use any of the given utility functions to perform
        augmentation.

        Args:
            input_anno: the input annotation to be replaced.

        Returns:
            A boolean value indicating if the augmentation
            was successful (True) or unsuccessful (False).
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entry:
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                For example, "ft.onto.base_ontology.Sentence".
        """
        return {"augment_entry": "ft.onto.base_ontology.Token"}
