import sys
from termcolor import colored
from forte.data.readers import ClassificationDatasetReader
from fortex.huggingface import ZeroShotClassifier
from forte.pipeline import Pipeline
from forte.common.configuration import Config
from forte.nltk import NLTKWordTokenizer, NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence


csv_path = "path_to_dataset/amazon_review_polarity_csv/test.csv"

pl = Pipeline()

# initialize labels
class_names = ["negative", "positive"]
index2class = dict(enumerate(class_names))

# initialize reader config
this_read_config = {"index2class": index2class}

# initialize model config
model_config = Config(
    {"candidate_labels": class_names}, ZeroShotClassifier().default_configs()
)

pl.set_reader(ClassificationDatasetReader(), config=this_read_config)
pl.add(NLTKSentenceSegmenter())
pl.add(NLTKWordTokenizer())
pl.add(ZeroShotClassifier(), config=model_config)
pl.initialize()


for pack in pl.process_dataset(csv_path):
    for sentence in pack.get(Sentence):
        if (
            input("Type n for the next sentence and its prediction: ").lower()
            == "n"
        ):
            sent_text = sentence.text
            print(colored("Sentence:", "red"), sent_text, "\n")
            print(colored("Prediction:", "blue"), sentence.classification)
        else:
            print("Exit the program due to unrecognized input")
            sys.exit()
