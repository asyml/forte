# Copyright 2020 The Forte Authors. All Rights Reserved.
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

import yaml

from termcolor import colored

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from ft.onto.base_ontology import Subword, Sentence, EntityMention

from examples.biobert_ner.transformers_processor import BERTTokenizer
from examples.biobert_ner.bio_ner_predictor import BioBERTNERPredictor

config = yaml.safe_load(open("config.yml", "r"))

config = Config(config, default_hparams=None)


def main():
    pl = Pipeline[DataPack]()
    pl.set_reader(StringReader())
    pl.add(NLTKSentenceSegmenter())
    pl.add(BERTTokenizer(), config=config.BERTTokenizer)
    pl.add(BioBERTNERPredictor(), config=config.BioBERTNERPredictor)
    pl.initialize()

    text = (
        "More than three-quarters of patients (77.5%) had comorbidities. "
        "Twenty-four isolates (60%) were associated with pneumonia, "
        "14 (35%) with upper respiratory tract infections, "
        "and 2 (5%) with bronchiolitis. "
        "The 3 patients who died of M pneumoniae pneumonia "
        "had other comorbidities. ")
    pack = pl.process(text)

    for sentence in pack.get(Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        # first method to get entry in a sentence
        subwords = [(subword.text, subword.ner) for subword in
                    pack.get(Subword, sentence)]
        entities = [(entity.text, entity.ner_type) for entity in
                    pack.get(EntityMention, sentence)]
        print(colored("Subwords:", 'red'), subwords, "\n")
        print(colored("EntityMentions:", 'red'), entities, "\n")

        input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == '__main__':
    main()
