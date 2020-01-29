# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nltk import word_tokenize, pos_tag, sent_tokenize, ne_chunk
from nltk.chunk import RegexpParser
from nltk.stem import WordNetLemmatizer

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import EntityMention, Token, Sentence, Phrase


__all__ = [
    "NLTKPOSTagger",
    "NLTKSentenceSegmenter",
    "NLTKWordTokenizer",
    "NLTKLemmatizer",
    "NLTKChunker",
    "NLTKNER",
]


class NLTKWordTokenizer(PackProcessor):
    r"""A wrapper of NLTK word tokenizer.
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
    r"""A wrapper of NLTK pos tagger.
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
                token.set_fields(pos=tag[1])


class NLTKLemmatizer(PackProcessor):
    r"""A wrapper of NLTK lemmatizer.
    """
    def __init__(self):
        super().__init__()
        self.token_component = None
        self.lemmatizer = WordNetLemmatizer()

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(input_pack.get(entry_type=Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            token_texts = [token.text for token in token_entries]
            token_pos = [penn2morphy(token.pos)  # type: ignore
                         for token in token_entries]
            lemmas = [self.lemmatizer.lemmatize(token_texts[i], token_pos[i])
                      for i in range(len(token_texts))]
            for token, lemma in zip(token_entries, lemmas):
                token.set_fields(lemma=lemma)


def penn2morphy(penntag: str) -> str:
    r"""Converts tags from Penn format to Morphy.
    """
    morphy_tag = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    if penntag[:2] in morphy_tag:
        return morphy_tag[penntag[:2]]
    else:
        return 'n'


class NLTKChunker(PackProcessor):
    r"""A wrapper of NLTK chunker.
    """
    def __init__(self):
        super().__init__()
        self.chunker = None
        self.token_component = None

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources, configs: HParams):
        self.chunker = RegexpParser(configs.pattern)

    @staticmethod
    def default_configs():
        r"""This defines a basic Hparams structure for NLTKChunker.
        """
        return {
            'pattern': 'NP: {<DT>?<JJ>*<NN>}',
        }

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(input_pack.get(entry_type=Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            tokens = [(token.text, token.pos) for token in token_entries]
            cs = self.chunker.parse(tokens)

            index = 0
            for chunk in cs:
                if hasattr(chunk, 'label'):
                    # For example:
                    # chunk: Tree('NP', [('This', 'DT'), ('tool', 'NN')])
                    begin_pos = token_entries[index].span.begin
                    end_pos = token_entries[index + len(chunk) - 1].span.end
                    phrase = Phrase(input_pack, begin_pos, end_pos)
                    kwargs_i = {"phrase_type": chunk.label()}
                    phrase.set_fields(**kwargs_i)
                    input_pack.add_or_get_entry(phrase)
                    index += len(chunk)
                else:
                    # For example:
                    # chunk: ('is', 'VBZ')
                    index += 1


class NLTKSentenceSegmenter(PackProcessor):
    r"""A wrapper of NLTK sentence tokenizer.
    """
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


class NLTKNER(PackProcessor):
    r"""A wrapper of NLTK NER.
    """
    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(input_pack.get(entry_type=Token,
                                                range_annotation=sentence,
                                                component=self.token_component))
            tokens = [(token.text, token.pos) for token in token_entries]
            ne_tree = ne_chunk(tokens)

            index = 0
            for chunk in ne_tree:
                if hasattr(chunk, 'label'):
                    # For example:
                    # chunk: Tree('GPE', [('New', 'NNP'), ('York', 'NNP')])
                    begin_pos = token_entries[index].span.begin
                    end_pos = token_entries[index + len(chunk) - 1].span.end
                    entity = EntityMention(input_pack, begin_pos, end_pos)
                    kwargs_i = {"ner_type": chunk.label()}
                    entity.set_fields(**kwargs_i)
                    input_pack.add_or_get_entry(entity)
                    index += len(chunk)
                else:
                    # For example:
                    # chunk: ('This', 'DT')
                    index += 1
