"""
Unit tests of NLTK processors.
"""

import unittest

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKLemmatizer, NLTKChunker
from ft.onto.base_ontology import Token, Sentence, Phrase


class TestNLTKSentenceSegmenter(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())

        self.nltk.initialize()

    def test_segmenter(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)
        for idx, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(sentence.text, sentences[idx])


class TestNLTKWordTokenizer(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.add_processor(NLTKWordTokenizer())

        self.nltk.initialize()

    def test_tokenizer(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        tokens = [["This", "tool", "is", "called", "Forte", "."],
                  ["The", "goal", "of", "this", "project", "to", "help", "you",
                   "build", "NLP", "pipelines", "."],
                  ["NLP", "has", "never", "been", "made", "this", "easy",
                   "before", "."]]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            for j, token in enumerate(
                    pack.get(entry_type=Token, range_annotation=sentence)):
                self.assertEqual(token.text, tokens[i][j])


class TestNLTKPOSTagger(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.add_processor(NLTKWordTokenizer())
        self.nltk.add_processor(NLTKPOSTagger())

        self.nltk.initialize()

    def test_pos_tagger(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        pos = [["DT", "NN", "VBZ", "VBN", "NNP", "."],
               ["DT", "NN", "IN", "DT", "NN", "TO", "VB", "PRP", "VB", "NNP",
                "NNS", "."],
               ["NNP", "VBZ", "RB", "VBN", "VBN", "DT", "JJ", "RB", "."]]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            for j, token in enumerate(
                    pack.get(entry_type=Token, range_annotation=sentence)):
                self.assertEqual(token.pos, pos[i][j])


class TestNLTKLemmatizer(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.add_processor(NLTKWordTokenizer())
        self.nltk.add_processor(NLTKPOSTagger())
        self.nltk.add_processor(NLTKLemmatizer())

        self.nltk.initialize()

    def test_lemmatizer(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        tokens = [["This", "tool", "be", "call", "Forte", "."],
                  ["The", "goal", "of", "this", "project", "to", "help", "you",
                   "build", "NLP", "pipeline", "."],
                  ["NLP", "have", "never", "be", "make", "this", "easy",
                   "before", "."]]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)
        for i, sentence in enumerate(pack.get(Sentence)):
            for j, token in enumerate(
                    pack.get(entry_type=Token, range_annotation=sentence)):
                self.assertEqual(token.lemma, tokens[i][j])


class TestNLTKChunker(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.add_processor(NLTKWordTokenizer())
        self.nltk.add_processor(NLTKPOSTagger())
        config = HParams({'pattern': 'NP: {<DT>?<JJ>*<NN>}'},
                         NLTKChunker.default_hparams())
        self.nltk.add_processor(NLTKChunker(), config=config)

        self.nltk.initialize()

    def test_chunker(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)

        entities_text = [x.text for x in pack.annotations if
                         isinstance(x, Phrase)]
        entities_type = [x.phrase_type for x in pack.annotations if
                         isinstance(x, Phrase)]

        self.assertEqual(entities_text, ['This tool', 'The goal',
                                         'this project'])
        self.assertEqual(entities_type, ['NP', 'NP', 'NP'])


if __name__ == "__main__":
    unittest.main()
