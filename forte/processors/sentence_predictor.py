from nltk.tokenize import sent_tokenize

from forte.data import DataPack
from forte.data.ontology import base_ontology
from forte.processors.base import PackProcessor, ProcessInfo

__all__ = [
    "NLTKSentenceSegmenter",
    "PeriodSentenceSegmenter"
]


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


class PeriodSentenceSegmenter(PackProcessor):
    """
    A dummy sentence segmenter which segments sentence only based on periods.
    Used for unit tests.
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

        begin_pos = 0
        while begin_pos < len(text):
            end_pos = min(text.find('.', begin_pos))
            if end_pos == -1:
                end_pos = len(text) - 1
            sentence_entry = self._ontology.Sentence(
                input_pack, begin_pos, end_pos + 1)
            input_pack.add_or_get_entry(sentence_entry)

            begin_pos = end_pos + 1
            while begin_pos < len(text) and text[begin_pos] == " ":
                begin_pos += 1
