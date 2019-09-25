from nltk.tokenize import word_tokenize

from forte.processors.base import PackProcessor, ProcessInfo
from forte.data import DataPack
from forte.data.ontology import base_ontology

__all__ = [
    "NLTKWordTokenizer",
]


class NLTKWordTokenizer(PackProcessor):

    def __init__(self):
        super().__init__()
        self.sentence_component = None
        self._ontology = base_ontology

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Sentence: ["span"]
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self._ontology.Token: ["span"]
        }
        return output_info

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(entry_type=self._ontology.Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin
            end_pos = 0
            for word in word_tokenize(sentence.text):
                begin_pos = sentence.text.find(word, end_pos)
                end_pos = begin_pos + len(word)
                token = self._ontology.Token(
                    input_pack,
                    begin_pos + offset, end_pos + offset
                )
                input_pack.add_or_get_entry(token)
