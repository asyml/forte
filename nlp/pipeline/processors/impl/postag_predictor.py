from nltk import pos_tag
from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack, BaseOntology

__all__ = [
    "NLTKPOSTagger"
]


class NLTKPOSTagger(PackProcessor):

    def __init__(self):
        super().__init__()

        self.ontology = BaseOntology  # should specify for each pipeline
        self.token_component = None

    def _process(self, input_pack: DataPack):
        # TODO: need to think about how to specify component, currently is
        #  getting all component
        for sentence in input_pack.get(self.ontology.Sentence):
            token_entries = list(input_pack.get(entry_type=self.ontology.Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            token_texts = [token.text for token in token_entries]
            taggings = pos_tag(token_texts)
            for token, tag in zip(token_entries, taggings):
                token.pos_tag = tag

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            ["pos_tag"],
            self.ontology.Token.__name__,
        )
