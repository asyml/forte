"""
Unit tests of NLTK processors.
"""

import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKLemmatizer, NLTKNER
from ft.onto.base_ontology import Token, Sentence, EntityMention


class TestNLTKSentenceSegmenter(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())

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


class TestNLTKNER(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.add_processor(NLTKWordTokenizer())
        self.nltk.add_processor(NLTKPOSTagger())
        self.nltk.add_processor(NLTKNER())

        self.nltk.initialize()

    def test_ner(self):
        sentences = ["This tool is called New   York.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.nltk.process(document)

        entities_entries = list(pack.get(entry_type=EntityMention))

        entities_text = [x.text for x in entities_entries]
        entities_type = [x.ner_type for x in entities_entries]

        self.assertEqual(entities_text, ['New   York', 'NLP', 'NLP'])
        self.assertEqual(entities_type, ['GPE', 'ORGANIZATION', 'ORGANIZATION'])


if __name__ == "__main__":
    unittest.main()
