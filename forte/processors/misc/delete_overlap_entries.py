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
"""A processor to delete overlap entries."""

__all__ = [
    "DeleteOverlapEntry",
]

from typing import List
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.data.ontology import Annotation
from forte.utils import get_class
from forte.common.exception import ProcessorConfigError


class DeleteOverlapEntry(PackProcessor):
    """
    A processor to delete overlap entries in a datapack.
    """
    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        self.resources = resources
        self.config = Config(configs, self.default_configs())
        if not self.config.entry_type:
            raise ProcessorConfigError(
                "Please specify an entity mention type!")
        self.entry_type = get_class(self.config.entry_type)

    def _process(self, input_pack: DataPack):
        entry_spans: List[Annotation] = []
        for entry in list(input_pack.get(self.entry_type)):
            current_span = entry.span
            if entry_spans and \
                self._is_overlap(entry_spans[-1], current_span):
                input_pack.delete_entry(entry)
            else:
                entry_spans.append(entry.span)

    def _is_overlap(self, interval1: Annotation, interval2: Annotation) -> bool:
        """
        Determine whether two intervals overlap with each other.
        """
        if interval2.begin < interval1.begin:
            interval1, interval2 = interval2, interval1
        return interval2.begin <= interval1.end

    @classmethod
    def default_configs(cls):
        configs = super().default_configs()
        configs.update({
            "entry_type": None,
        })
        return configs
