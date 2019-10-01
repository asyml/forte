"""
This file contains examples of batch processor implementations, which basically
create entries arbitrarily. The processors here are useful as placeholders and
test cases.

"""
from typing import Dict, Optional

import numpy as np

from forte.data import DataPack
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.ontology import relation_ontology
from forte.data.ontology.relation_ontology import EntityMention
from forte.processors.base import BatchProcessor, ProcessInfo

__all__ = [
    "DummyRelationExtractor",
]


class DummyRelationExtractor(BatchProcessor):
    """
    A dummy relation extractor
    """

    def __init__(self) -> None:
        super().__init__()
        self._ontology = relation_ontology
        self.define_context()

        self.batch_size = 4
        self.batcher = self.define_batcher()

    def define_batcher(self) -> ProcessingBatcher:
        # pylint: disable=no-self-use
        return FixedSizeDataPackBatcher()

    def define_context(self):
        self.context_type = self._ontology.Sentence

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Token: [],
            self._ontology.EntityMention: {
                "fields": ["ner_type", "tid"]
            }
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self._ontology.RelationLink:
                ["parent", "child", "rel_type"]
        }
        return output_info

    def predict(self, data_batch: Dict):  # pylint: disable=no-self-use
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
        """Add corresponding fields to data_pack"""
        if output_dict is None:
            return

        for i in range(len(output_dict["RelationLink"]["parent.tid"])):
            for j in range(len(output_dict["RelationLink"]["parent.tid"][i])):
                link = self._ontology.RelationLink(data_pack)
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["parent.tid"][i][j])
                link.set_parent(parent)
                child: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["child.tid"][i][j])
                link.set_child(child)
                data_pack.add_or_get_entry(link)
