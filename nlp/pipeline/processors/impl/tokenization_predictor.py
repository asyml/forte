from typing import Dict
import numpy as np
from nltk.tokenize import word_tokenize
from nlp.pipeline.processors import Predictor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.readers import CoNLL03Ontology

__all__ = [
    "NLTKWordTokenizer",
]


class NLTKWordTokenizer(Predictor):

    def __init__(self):
        super().__init__()

        self.context_type = "sentence"
        self.batch_size = 4
        self.ontology = CoNLL03Ontology  # should specify for each pipeline

    def predict(self, data_batch: Dict):
        sentences = data_batch["context"]
        offsets = data_batch["offset"]
        pred = {
            "Token": {
                "begin": [],
                "end": [],
            }
        }

        for sent, offset in zip(sentences, offsets):
            begins = []
            ends = []
            end_pos = 0  # the ending position of the previous word
            words = word_tokenize(sent)
            for word in words:
                begin_pos = sent.find(word, end_pos)
                end_pos = begin_pos + len(word)
                begins.append(begin_pos + offset)
                ends.append(end_pos + offset)

            pred["Token"]["begin"].append(np.array(begins))
            pred["Token"]["end"].append(np.array(ends))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Dict = None) -> None:

        if output_dict is None:
            return

        for i in range(len(output_dict["Token"]["begin"])):
            for j in range(len(output_dict["Token"]["begin"][i])):
                token = self.ontology.Token(
                    self.component_name,
                    output_dict["Token"]["begin"][i][j],
                    output_dict["Token"]["end"][i][j])
                data_pack.add_entry(token)

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            [],
            self.ontology.Token.__name__,
            self.component_name,
        )
