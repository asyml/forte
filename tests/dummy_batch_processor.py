# Copyright 2019 The Forte Authors. All Rights Reserved.
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
This file contains examples of batch processor implementations, which basically
create entries arbitrarily. The processors here are useful as placeholders and
test cases.
"""
from typing import Dict, Optional, Type

import numpy as np

from forte.data.data_pack import DataPack
from forte.data.types import DataRequest
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.processors.base import BatchProcessor, FixedSizeBatchProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink

__all__ = [
    "DummyRelationExtractor",
    "DummmyFixedSizeBatchProcessor",
]


class DummyRelationExtractor(BatchProcessor):
    r"""A dummy relation extractor.

    Note that to use :class:`DummyRelationExtractor`, the :attr:`ontology` of
    :class:`Pipeline` must be an ontology that includes
    ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()
        self.batcher = self.define_batcher()

    def define_batcher(self) -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()

    def define_context(self) -> Type[Sentence]:
        return Sentence

    def _define_input_info(self) -> DataRequest:
        input_info: DataRequest = {
            Token: [],
            EntityMention: {"fields": ["ner_type", "tid"]}
        }
        return input_info

    def predict(self, data_batch: Dict):
        entities_span = data_batch["EntityMention"]["span"]
        entities_tid = data_batch["EntityMention"]["tid"]

        pred: Dict = {
            "RelationLink": {
                "parent.tid": [],
                "child.tid": [],
                "rel_type": [],
            }
        }
        for tid, entity in zip(entities_tid, entities_span):
            parent = []
            child = []
            rel_type = []

            entity_num = len(entity)
            for i in range(entity_num):
                for j in range(i + 1, entity_num):
                    parent.append(tid[i])
                    child.append(tid[j])
                    rel_type.append("dummy_relation")

            pred["RelationLink"]["parent.tid"].append(
                np.array(parent))
            pred["RelationLink"]["child.tid"].append(
                np.array(child))
            pred["RelationLink"]["rel_type"].append(
                np.array(rel_type))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        r"""Add corresponding fields to data_pack"""
        if output_dict is None:
            return

        for i in range(len(output_dict["RelationLink"]["parent.tid"])):
            for j in range(len(output_dict["RelationLink"]["parent.tid"][i])):
                link = RelationLink(data_pack)
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["parent.tid"][i][j])
                link.set_parent(parent)
                child: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["child.tid"][i][j])
                link.set_child(child)
                data_pack.add_or_get_entry(link)

    @staticmethod
    def default_configs():
        return {
            "batcher": {"batch_size": 10}
        }


class DummmyFixedSizeBatchProcessor(FixedSizeBatchProcessor):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.batcher = self.define_batcher()

    def define_context(self) -> Type[Sentence]:
        return Sentence

    def _define_input_info(self) -> DataRequest:
        return {}

    def predict(self, data_batch: Dict):
        # track the number of times `predict` was called
        self.counter += 1
        return data_batch

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        r"""Add corresponding fields to data_pack"""
        pass

    @staticmethod
    def default_configs():
        return {
            "batcher": {"batch_size": 10}
        }
