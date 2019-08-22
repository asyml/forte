import os
import stanfordnlp
import forte.data.ontology.stanfordnlp_ontology as ontology
from forte.processors.base import PackProcessor
from forte.data import DataPack

__all__ = [
    "StandfordNLPProcessor",
]


class StandfordNLPProcessor(PackProcessor):

    def __init__(self, models_path: str = os.getcwd()):
        super().__init__()
        self._ontology = ontology
        self.processors = ""
        self.nlp = None
        self.MODELS_DIR = models_path
        self.lang = 'en'  # English is default
        self.define_input_info()
        self.define_output_info()

    def set_up(self):
        if os.path.exists(self.MODELS_DIR):
            print("Using models at {}".format(self.MODELS_DIR))
        else:
            stanfordnlp.download(self.lang, self.MODELS_DIR)

    def define_input_info(self):
        self.input_info = {
            self._ontology.Document: ["span"]
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.Token: ["span", "pos_tag", "upos", "lemma",
                                   "xpos", "dependency_relation"],
            self._ontology.Sentence: ["span"]
        }

    def initialize(self, configs):
        self.processors = configs['processors']
        self.lang = configs['lang']
        self.set_up()
        self.nlp = stanfordnlp.Pipeline(**configs)

    def _process(self, input_pack: DataPack):

        doc = input_pack.text
        end_pos = 0

        # sentence parsing
        sentences = self.nlp(doc).sentences  # type: ignore

        # Iterating through stanfordnlp sentence objects
        for sentence in sentences:
            begin_pos = doc.find(sentence.words[0].text, end_pos)
            end_pos = doc.find(sentence.words[-1].text, begin_pos) \
                      + len(sentence.words[-1].text)
            sentence_entry = self._ontology.Sentence(begin_pos, end_pos)
            input_pack.add_or_get_entry(sentence_entry)

            if "tokenize" in self.processors:
                offset = sentence_entry.span.begin
                end_pos_word = 0

                # Iterating through stanfordnlp word objects
                for word in sentence.words:
                    begin_pos_word = sentence_entry.text.\
                        find(word.text, end_pos_word)
                    end_pos_word = begin_pos_word + len(word.text)
                    token = self._ontology.Token(
                        begin_pos_word + offset, end_pos_word + offset
                    )

                    if "pos" in self.processors:
                        token.pos_tag = word.pos
                        token.upos = word.upos

                    if "lemma" in self.processors:
                        token.lemma = word.lemma
                        token.xpos = word.xpos

                    if "depparse" in self.processors:
                        token.dependency_relation = word.dependency_relation

                    input_pack.add_or_get_entry(token)
