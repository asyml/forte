from typing import Dict
import numpy as np
from nltk.tokenize import sent_tokenize
from nlp.pipeline.processors import Predictor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.readers import CoNLL03Ontology

__all__ = [
    "NLTKSentenceSegmenter",
]


class NLTKSentenceSegmenter(Predictor):

    def __init__(self):
        super().__init__()

        self.context_type = "document"
        self.batch_size = 4
        self.ontology = CoNLL03Ontology  # should specify for each pipeline

    def predict(self, data_batch: Dict):
        docs = data_batch["context"]
        offsets = data_batch["offset"]
        pred = {
            "Sentence": {
                "begin": [],
                "end": [],
            }
        }

        for doc, offset in zip(docs, offsets):
            # texts = []
            begins = []
            ends = []
            end_pos = 0  # the ending position of the previous sentence
            paragraphs = [p for p in doc.split('\n') if p]
            for paragraph in paragraphs:
                sentences = sent_tokenize(paragraph)
                for sent in sentences:
                    begin_pos = doc.find(sent, end_pos)
                    end_pos = begin_pos + len(sent)
                    begins.append(begin_pos+offset)
                    ends.append(end_pos+offset)

            pred["Sentence"]["begin"].append(np.array(begins))
            pred["Sentence"]["end"].append(np.array(ends))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Dict = None) -> None:

        if output_dict is None: return

        for i in range(len(output_dict["Sentence"]["begin"])):
            for j in range(len(output_dict["Sentence"]["begin"][i])):
                sent = self.ontology.Sentence(
                    self.component_name,
                    output_dict["Sentence"]["begin"][i][j],
                    output_dict["Sentence"]["end"][i][j])
                data_pack.add_entry(sent)

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            [],
            self.ontology.Sentence.__name__,
            self.component_name,
        )