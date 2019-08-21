import stanfordnlp
import forte.data.ontology.stanfordnlp_ontology as ontology
from forte.processors.base import PackProcessor
from forte.data import PackType
from forte.common.resources import Resources

__all__ = [
    "StandfordNLPProcessor",
]

class StandfordNLPProcessor(PackProcessor):

    def __init__(self):
        super().__init__()
        self.sentence_component = None
        self._ontology = ontology
        self.processors = ""

    def define_input_info(self):
        self.input_info = {
            self._ontology.Document: ["span"]
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.Token: ["span"],
            self._ontology.Sentence: ["span"]
        }

    def initialize(self, configs, resource: Resources):
        self.processors = configs
        self.nlp = stanfordnlp.Pipeline(self.processors)

    def _process(self, input_pack: PackType):

        text = input_pack.text
        end_pos = 0

        # sentence parsing
        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            sentences = self.nlp(paragraph).sentences
            for sentence in sentences:
                begin_pos = text.find(sentence.words[0].text, end_pos)
                end_pos = text.find(sentence.words[-1].text, begin_pos) \
                          + len(sentence.words[-1].text)
                sentence_entry = self._ontology.Sentence(begin_pos, end_pos)
                input_pack.add_or_get_entry(sentence_entry)

                if "tokenize" in self.processors:
                    offset = sentence_entry.span.begin
                    end_pos_word = 0
                    for word in sentence.words:
                        begin_pos_word = sentence_entry.text.find(word.text, end_pos_word)
                        end_pos_word = begin_pos_word + len(word.text)
                        token = self._ontology.Token(
                            begin_pos_word + offset, end_pos_word + offset
                        )

                        token.__setattr__(text, word.text)

                        if "pos" in self.processors:
                            token.pos_tag = word.pos
                            token.upos = word.upos

                        if "lemma" in self.processors:
                            token.lemma = word.lemma
                            token.xpos = word.xpos

                        if "depparse" in self.processors:
                            token.dependency_relation = word.dependency_relation

                        input_pack.add_or_get_entry(token)
