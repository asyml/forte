import sys
from termcolor import colored
from forte.data.readers import ClassificationDatasetReader
from fortex.huggingface import ZeroShotClassifier
from forte.pipeline import Pipeline
from fortex.nltk import NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence


csv_path = "data_samples/amazon_review_polarity_csv/sample.csv"
pl = Pipeline()

# initialize labels
class_names = ["negative", "positive"]
index2class = dict(enumerate(class_names))
pl.set_reader(
    ClassificationDatasetReader(), config={"index2class": index2class}
)
pl.add(NLTKSentenceSegmenter())
pl.add(ZeroShotClassifier(), config={"candidate_labels": class_names})
pl.initialize()


for pack in pl.process_dataset(csv_path):
    for sent in pack.get(Sentence):
        if (
            input("Type n for the next documentation and its prediction: ").lower()
            == "n"
        ):
            sent_text = sent.text
            print(colored("Sentence:", "red"), sent_text, "\n")
            print(colored("Prediction:", "blue"), sent.classification)
        else:
            print("Exit the program due to unrecognized input")
            sys.exit()
