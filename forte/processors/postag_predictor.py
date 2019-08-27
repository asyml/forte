from nltk import pos_tag

from forte.data import DataPack
from forte.data.ontology import base_ontology
from forte.processors.base import PackProcessor, ProcessInfo

__all__ = [
    "NLTKPOSTagger"
]


class NLTKPOSTagger(PackProcessor):

    def __init__(self):
        super().__init__()
        self.token_component = None
        self._ontology = base_ontology

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Sentence: ["span"],
            self._ontology.Token: {
                "fields": ["span"],
                "component": self.token_component
            }
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self._ontology.Token: {
                "component": self.token_component,
                "fields": ["pos_tag"]
            }
        }
        return output_info

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(self._ontology.Sentence):
            token_entries = list(input_pack.get(entry_type=self._ontology.Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            token_texts = [token.text for token in token_entries]
            taggings = pos_tag(token_texts)
            for token, tag in zip(token_entries, taggings):
                token.pos_tag = tag[1]
