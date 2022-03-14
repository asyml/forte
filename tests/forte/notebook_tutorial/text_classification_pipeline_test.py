from forte.data.readers import ClassificationDatasetReader
from fortex.huggingface import ZeroShotClassifier
from forte.pipeline import Pipeline
from fortex.nltk import NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence

import os
import unittest

from forte.pipeline import Pipeline


class TestClassificationPipeline(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )

        self.csv_path = "data_samples/amazon_review_polarity_csv/sample.csv"
        self.pl = Pipeline()

        # initialize labels
        class_names = ["negative", "positive"]
        index2class = dict(enumerate(class_names))
        self.pl.set_reader(
            ClassificationDatasetReader(), config={"index2class": index2class}
        )
        self.pl.add(NLTKSentenceSegmenter())
        self.pl.add(
            ZeroShotClassifier(), config={"candidate_labels": class_names}
        )
        self.pl.initialize()

        self.sents = [
            "One of the best game music soundtracks - for a game I didn't really play\nDespite the fact that I have only played a small portion of the game, the music I heard (plus the connection to Chrono Trigger which was great as well) led me to purchase the soundtrack, and it remains one of my favorite albums.",
            "There is an incredible mix of fun, epic, and emotional songs.",
            "Those sad and beautiful tracks I especially like, as there's not too many of those kinds of songs in my other video game soundtracks.",
            "I must admit that one of the songs (Life-A Distant Promise) has brought tears to my eyes on many occasions.My one complaint about this soundtrack is that they use guitar fretting effects in many of the songs, which I find distracting.",
            "But even if those weren't included I would still consider the collection worth it.",
        ]

    def test_get(self):
        i = 0
        for pack in self.pl.process_dataset(self.csv_path):
            for sent in pack.get(Sentence):
                sent_text = sent.text
                self.assertEqual(self.sents[i], sent_text)
                self.assertTrue(sent.classification["positive"] > 0)
                self.assertTrue(sent.classification["negative"] > 0)
                i += 1
            break
