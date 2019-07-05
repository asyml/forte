from typing import Dict
import nltk
import numpy as np
from nlp.pipeline.processors import Predictor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.readers import CoNLL03Ontology


class NLTKPOSTagger(Predictor):

    def __init__(self):
        super().__init__()

        self.context_type = "sentence"
        self.annotation_types = {
            "Token": [],
        }
        self.batch_size = 4
        self.ontology = CoNLL03Ontology  # should specify for each pipeline

    def predict(self, data_batch: Dict):

        pred = {
            "Token": {
                "pos_tag": [],
                "tid": [],
            }
        }

        for word_list, tid_list in zip(data_batch["Token"]["text"],
                                       data_batch["Token"]["tid"]):
            pos_tags = []
            taggings = nltk.pos_tag(word_list)

            for tag, tid in zip(taggings, tid_list):
                pos_tags.append(tag[1])

            pred["Token"]["pos_tag"].append(np.array(pos_tags))
            pred["Token"]["tid"].append(np.array(tid_list))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Dict = None) -> None:

        if output_dict is None: return

        for i in range(len(output_dict["Token"]["tid"])):
            for j in range(len(output_dict["Token"]["tid"][i])):
                tid = output_dict["Token"]["tid"][i][j]
                pos_tag = output_dict["Token"]["pos_tag"][i][j]
                data_pack.index.entry_index[tid].pos_tag = pos_tag

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            ["pos_tag", "chunk_tag"],  # shouldn't have chunk tag here
            self.ontology.Token.__name__,
        )
