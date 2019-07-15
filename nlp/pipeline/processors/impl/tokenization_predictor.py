from nltk.tokenize import word_tokenize
from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack, BaseOntology

__all__ = [
    "NLTKWordTokenizer",
]


class NLTKWordTokenizer(PackProcessor):

    def __init__(self):
        super().__init__()
        self.ontology = BaseOntology  # should specify for each pipeline
        self.sentence_component = None

    def _process(self, input_pack: DataPack):
        # TODO: need to think about how to specify component
        for sentence in input_pack.get(entry_type=self.ontology.Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin
            end_pos = 0
            for word in word_tokenize(sentence.text):
                begin_pos = sentence.text.find(word, end_pos)
                end_pos = begin_pos + len(word)
                token = self.ontology.Token(
                    self.component_name, begin_pos + offset, end_pos + offset)
                input_pack.add_entry(token)

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            ["span"],
            self.ontology.Token.__name__,
            self.component_name,
        )
