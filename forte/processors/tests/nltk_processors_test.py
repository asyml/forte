"""
Unit tests of NLTK processors.
"""

import unittest

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKLemmatizer, NLTKChunker, NLTKNER
from ft.onto.base_ontology import Token, Sentence, Phrase, EntityMention


class TestNLTKSentenceSegmenter(unittest.TestCase):

    def setUp(self):
        self.nltk = Pipeline()
        self.nltk.set_reader(StringReader())
        self.nltk.add_processor(NLTKSentenceSegmenter())
        self.nltk.initialize()

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

        phrase_entries = list(pack.get(entry_type=Phrase))

        entities_text = [x.text for x in phrase_entries]
        entities_type = [x.phrase_type for x in phrase_entries]

        self.assertEqual(entities_text, ['This tool', 'The goal',
                                         'this project'])
        self.assertEqual(entities_type, ['NP', 'NP', 'NP'])


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
