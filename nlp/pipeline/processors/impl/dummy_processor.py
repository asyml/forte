from typing import Dict, Optional

import numpy as np

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import relation_ontology
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.processors.batch_processor import BatchProcessor

__all__ = [
    "DummyRelationExtractor",
]


class DummyRelationExtractor(BatchProcessor):
    """
    A dummy relation extractor
    """

    def __init__(self) -> None:
        super().__init__()
        self.ontology = relation_ontology  # the output should be in this onto

        self.context_type = "sentence"
        self.input_info = {
            base_ontology.Token: [],
            base_ontology.EntityMention: {
                "fields": ["ner_type", "tid"],
            }
        }
        self.output_info = {
            self.ontology.RelationLink:  # type: ignore
                ["parent", "child", "rel_type"]
        }

        self.batch_size = 4
        self.initialize_batcher()

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
                link = self.ontology.RelationLink(
                    component=self.component_name)
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent = data_pack.index.entry_index[
                    output_dict["RelationLink"]["parent.tid"][i][j]]
                link.set_parent(parent)
                child = data_pack.index.entry_index[
                    output_dict["RelationLink"]["child.tid"][i][j]]
                link.set_child(child)
                data_pack.add_or_get_entry(link)
