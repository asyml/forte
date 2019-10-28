from forte.data import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence

__all__ = [
    "PeriodSentenceSegmenter"
]


class PeriodSentenceSegmenter(PackProcessor):
    """
    A dummy sentence segmenter which segments sentence only based on periods.
    Used for unit tests.
    """
    def _process(self, input_pack: DataPack):
        # pylint: disable=no-self-use
        text = input_pack.text

        begin_pos = 0
        while begin_pos < len(text):
            end_pos = min(text.find('.', begin_pos))
            if end_pos == -1:
                end_pos = len(text) - 1
            sentence_entry = Sentence(input_pack, begin_pos, end_pos + 1)
            input_pack.add_or_get_entry(sentence_entry)

            begin_pos = end_pos + 1
            while begin_pos < len(text) and text[begin_pos] == " ":
                begin_pos += 1
