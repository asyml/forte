from typing import Dict, Optional

import numpy as np

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import relation_ontology
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.processors import BatchProcessor

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
        self.define_input_info()
        self.define_output_info()

        self.context_type = base_ontology.Sentence

        self.batch_size = 4
        self.batcher = self.initialize_batcher()

    def define_input_info(self):
        self.input_info = {
            base_ontology.Token: [],
            base_ontology.EntityMention: {
                "fields": ["ner_type", "tid"],
            }
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.RelationLink:
                ["parent", "child", "rel_type"]
        }

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
            ner_type = []

            entity_num = len(entity)
            for i in range(entity_num):
                for j in range(i + 1, entity_num):
                    parent.append(tid[i])
                    child.append(tid[j])
                    ner_type.append("dummy_relation")

            pred["RelationLink"]["parent.tid"].append(
                np.array(parent))
            pred["RelationLink"]["child.tid"].append(
                np.array(child))
            pred["RelationLink"]["rel_type"].append(
                np.array(ner_type))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        """Add corresponding fields to data_pack"""
        if output_dict is None:
            return

        for i in range(len(output_dict["RelationLink"]["parent.tid"])):
            for j in range(len(output_dict["RelationLink"]["parent.tid"][i])):
                link = self._ontology.RelationLink()
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent = data_pack.get_entry_by_id(
                    output_dict["RelationLink"]["parent.tid"][i][j])
                link.set_parent(parent)
                child = data_pack.get_entry_by_id(
                    output_dict["RelationLink"]["child.tid"][i][j])
                link.set_child(child)
                data_pack.add_or_get_entry(link)
