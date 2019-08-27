import stanfordnlp
import forte.data.ontology.stanfordnlp_ontology as ontology
from forte.processors.base import PackProcessor, ProcessInfo
from forte.data import DataPack
from forte.common.resources import Resources

__all__ = [
    "StandfordNLPProcessor",
]


class StandfordNLPProcessor(PackProcessor):
    def __init__(self, models_path: str):
        super().__init__()
        self._ontology = ontology
        self.processors = ""
        self.nlp = None
        self.MODELS_DIR = models_path
        self.lang = 'en'  # English is default

    def set_up(self):
        stanfordnlp.download(self.lang, self.MODELS_DIR)

    # pylint: disable=unused-argument
    def initialize(self, configs: dict, resource: Resources):
        self.processors = configs['processors']
        self.lang = configs['lang']
        self.set_up()
        self.nlp = stanfordnlp.Pipeline(**configs, models_dir=self.MODELS_DIR)

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Document: ["span"]
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        token_outputs = ['span']
        if 'pos' in self.processors:
            token_outputs.append('pos_tag')
            token_outputs.append('upos')
            token_outputs.append('xpos')
        if 'lemma' in self.processors:
            token_outputs.append('lemma')
        if 'depparse' in self.processors:
            token_outputs.append('dependency_relation')

        output_info: ProcessInfo = {
            self._ontology.Token: token_outputs,
            self._ontology.Sentence: ["span"]
        }

        return output_info

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
                    begin_pos_word = sentence_entry.text. \
                        find(word.text, end_pos_word)
                    end_pos_word = begin_pos_word + len(word.text)
                    token = self._ontology.Token(
                        begin_pos_word + offset, end_pos_word + offset
                    )

                    if "pos" in self.processors:
                        token.pos_tag = word.pos
                        token.upos = word.upos
                        token.xpos = word.xpos

                    if "lemma" in self.processors:
                        token.lemma = word.lemma

                    if "depparse" in self.processors:
                        token.dependency_relation = word.dependency_relation

                    input_pack.add_or_get_entry(token)
