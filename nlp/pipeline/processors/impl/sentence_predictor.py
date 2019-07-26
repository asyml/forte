from nltk.tokenize import sent_tokenize

from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology

__all__ = [
    "NLTKSentenceSegmenter",
    "PeriodSentenceSegmenter"
]


class NLTKSentenceSegmenter(PackProcessor):

    def __init__(self):
        super().__init__()

        self.ontology = base_ontology  # should specify for each pipeline

        self.output_info = {
            self.ontology.Sentence: ["span"]
        }

    def _process(self, input_pack: DataPack):
        text = input_pack.text

        end_pos = 0
        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            for sentence_text in sentences:
                begin_pos = text.find(sentence_text, end_pos)
                end_pos = begin_pos + len(sentence_text)
                sentence_entry = self.ontology.Sentence(begin_pos, end_pos)
                input_pack.add_or_get_entry(sentence_entry)


class PeriodSentenceSegmenter(PackProcessor):
    """
    A dummy sentence segmenter which segments sentence only based on periods.
    Used for unit tests.
    """
    def __init__(self):
        super().__init__()

        self.ontology = base_ontology
        self.output_info = {
            self.ontology.Sentence: ["span"]
        }

    def _process(self, input_pack: DataPack):
        text = input_pack.text

        begin_pos = 0
        while begin_pos < len(text):
            end_pos = min(text.find('.', begin_pos))
            if end_pos == -1:
                end_pos = len(text)-1
            sentence_entry = self.ontology.Sentence(begin_pos, end_pos+1)
            input_pack.add_or_get_entry(sentence_entry)

            begin_pos = end_pos+1
            while begin_pos < len(text) and text[begin_pos] == " ":
                begin_pos += 1
