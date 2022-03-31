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
"""A processor to delete overlapping annotations."""

__all__ = [
    "DeleteOverlapEntry",
]

from typing import List

from forte.common import Resources
from forte.common.configuration import Config
from forte.common.exception import ProcessorConfigError
from forte.data import Span
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.base import PackProcessor
from forte.utils import get_class


class DeleteOverlapEntry(PackProcessor):
    """
    A processor to delete overlapping annotations in a data pack. When
    overlapping, the first annotation (based on the iteration order) will be
    kept and the rest of them will be removed.
    """

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if not self.configs.entry_type:
            raise ProcessorConfigError("Please specify an entity mention type!")

        self.entry_type = get_class(self.configs.entry_type)

        if not issubclass(self.entry_type, Annotation):
            raise AttributeError(
                f"The entry type to delete [{self.entry_type}] "
                f"is not a sub-class of "
                f"'forte.data.ontology.top.Annotation' class."
            )

    def _process(self, input_pack: DataPack):
        entry_spans: List[Span] = []
        entries: List[Annotation] = list(input_pack.get(self.entry_type))
        for entry in entries:
            current_span = entry.span
            if entry_spans and self._is_overlap(entry_spans[-1], current_span):
                input_pack.delete_entry(entry)
            else:
                entry_spans.append(current_span)

    def _is_overlap(self, interval1: Span, interval2: Span) -> bool:
        """
        Determine whether two intervals overlap with each other.
        """
        if interval2.begin < interval1.begin:
            interval1, interval2 = interval2, interval1
        return interval2.begin <= interval1.end

    @classmethod
    def default_configs(cls):
        """
        The ``entry_type`` config determines which type of annotation to be
        checked for duplication. This value should be the name of a class that
        is sub-class for :class:`~forte.data.ontology.top.Annotation`.
        Otherwise a `ValueError` will be raised.

        Returns:
            None.
        """

        return {
            "entry_type": None,
        }
