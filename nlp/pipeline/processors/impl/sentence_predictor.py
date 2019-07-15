from nltk.tokenize import sent_tokenize
from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack, BaseOntology

__all__ = [
    "NLTKSentenceSegmenter",
]


class NLTKSentenceSegmenter(PackProcessor):

    def __init__(self):
        super().__init__()
        self.ontology = BaseOntology  # should specify for each pipeline

    def _process(self, input_pack: DataPack):
        text = input_pack.text

        end_pos = 0
        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            for sentence_text in sentences:
                begin_pos = text.find(sentence_text, end_pos)
                end_pos = begin_pos + len(sentence_text)
                sentence_entry = self.ontology.Sentence(self.component_name,
                                                        begin_pos,
                                                        end_pos)
                input_pack.add_entry(sentence_entry)

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            ["span"],
            self.ontology.Sentence.__name__,
            self.component_name,
        )

