from nltk import word_tokenize, pos_tag, sent_tokenize

from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Token, Sentence


class NLTKWordTokenizer(PackProcessor):
    """
    A wrapper of NLTK word tokenizer.
    """
    def __init__(self):
        super().__init__()
        self.sentence_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(entry_type=Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin
            end_pos = 0
            for word in word_tokenize(sentence.text):
                begin_pos = sentence.text.find(word, end_pos)
                end_pos = begin_pos + len(word)
                token = Token(input_pack, begin_pos + offset, end_pos + offset)
                input_pack.add_or_get_entry(token)


class NLTKPOSTagger(PackProcessor):
    """
    A wrapper of NLTK pos tagger.
    """
    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(input_pack.get(entry_type=Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            token_texts = [token.text for token in token_entries]
            taggings = pos_tag(token_texts)
            for token, tag in zip(token_entries, taggings):
                token.pos = tag[1]


class NLTKSentenceSegmenter(PackProcessor):
    """
    A wrapper of NLTK sentence tokenizer.
    """
    # pylint: disable=no-self-use
    def _process(self, input_pack: DataPack):
        text = input_pack.text
        end_pos = 0
        paragraphs = [p for p in text.split('\n') if p]
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            for sentence_text in sentences:
                begin_pos = text.find(sentence_text, end_pos)
                end_pos = begin_pos + len(sentence_text)
                sentence_entry = Sentence(input_pack, begin_pos, end_pos)
                input_pack.add_or_get_entry(sentence_entry)
