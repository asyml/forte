from nltk.tokenize import word_tokenize

from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology

__all__ = [
    "NLTKWordTokenizer",
]


class NLTKWordTokenizer(PackProcessor):

    def __init__(self):
        super().__init__()
        self.sentence_component = None
        self._ontology = base_ontology
        self.define_input_info()
        self.define_output_info()

    def define_input_info(self):
        self.input_info = {
            base_ontology.Sentence: ["span"]
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.Token: ["span"]
        }

    def _process(self, input_pack: DataPack):
        # TODO: need to think about how to specify component
        for sentence in input_pack.get(entry_type=self._ontology.Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin
            end_pos = 0
            for word in word_tokenize(sentence.text):
                begin_pos = sentence.text.find(word, end_pos)
                end_pos = begin_pos + len(word)
                token = self._ontology.Token(
                    begin_pos + offset, end_pos + offset
                )
                input_pack.add_or_get_entry(token)
