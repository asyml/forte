from nltk import word_tokenize, pos_tag, sent_tokenize

from forte.data.data_pack import DataPack
from forte.data.ontology import base_ontology
from forte.processors.base import ProcessInfo
from forte.processors.base import PackProcessor


class NLTKWordTokenizer(PackProcessor):
    """
    A wrapper of NLTK word tokenizer.
    """
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


class NLTKPOSTagger(PackProcessor):
    """
    A wrapper of NLTK pos tagger.
    """
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


class NLTKSentenceSegmenter(PackProcessor):
    """
    A wrapper of NLTK sentence tokenizer.
    """
    def __init__(self):
        super().__init__()
        self._ontology = base_ontology

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Document: ["span"]
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {
            self._ontology.Sentence: ["span"]
        }
        return output_info

    def _process(self, input_pack: DataPack):
        text = input_pack.text
        end_pos = 0
        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            for sentence_text in sentences:
                begin_pos = text.find(sentence_text, end_pos)
                end_pos = begin_pos + len(sentence_text)
                sentence_entry = self._ontology.Sentence(
                    input_pack, begin_pos, end_pos)
                input_pack.add_or_get_entry(sentence_entry)
