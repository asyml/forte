from nltk import pos_tag

from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.processors.pack_processor import PackProcessor

__all__ = [
    "NLTKPOSTagger"
]


class NLTKPOSTagger(PackProcessor):

    def __init__(self):
        super().__init__()
        self.token_component = None
        self._ontology = base_ontology
        self.define_input_info()
        self.define_output_info()

    def define_input_info(self):
        self.input_info = {
            self._ontology.Sentence: ["span"],
            self._ontology.Token: {
                "fields": ["span"],
                "component": self.token_component
            }
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.Token: {
                "component": self.token_component,
                "fields": ["pos_tag"]
            }
        }

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(self._ontology.Sentence):
            token_entries = list(input_pack.get(entry_type=self._ontology.Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            token_texts = [token.text for token in token_entries]
            taggings = pos_tag(token_texts)
            for token, tag in zip(token_entries, taggings):
                token.pos_tag = tag[1]
