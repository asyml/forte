# Copyright 2022 The Forte Authors. All Rights Reserved.
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
            input(
                "Type n for the next documentation and its prediction: "
            ).lower()
            == "n"
        ):
            sent_text = sent.text
            print(colored("Sentence:", "red"), sent_text, "\n")
            print(colored("Prediction:", "blue"), sent.classification)
        else:
            print("Exit the program due to unrecognized input")
            sys.exit()
