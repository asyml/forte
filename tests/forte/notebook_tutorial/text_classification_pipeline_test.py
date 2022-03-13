import sys
from termcolor import colored
from forte.data.readers import ClassificationDatasetReader
from fortex.huggingface import ZeroShotClassifier
from forte.pipeline import Pipeline
from fortex.nltk import NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence

import os
import unittest

from forte.data.data_pack import DataPack
from forte.data.readers import ConllUDReader
from forte.pipeline import Pipeline
from forte.processors.misc import AttributeMasker
from ft.onto.base_ontology import Token
import os


class TestHandlingStructuedData(unittest.TestCase):
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

    def test_get(self):

        for pack in self.pl.process_dataset(self.addClassCleanupcsv_path):
            for sent in pack.get(Sentence):
                sent_text = sent.text
                print(colored("Sentence:", "red"), sent_text, "\n")
                print(colored("Prediction:", "blue"), sent.classification)
