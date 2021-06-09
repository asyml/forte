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

import argparse

import yaml
from termcolor import colored

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.readers import PlainTextReader
from forte.pipeline import Pipeline
from forte.processors.nlp import (CoNLLNERPredictor, SRLPredictor)
from forte.nltk import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter
from ft.onto.base_ontology import Token, Sentence, EntityMention, \
    PredicateLink, PredicateArgument, PredicateMention

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, default="data/",
                    help="Data directory to read the text files from")


def main(dataset_dir: str):
    config = yaml.safe_load(open("config.yml", "r"))
    config = Config(config, default_hparams=None)

    pl = Pipeline[DataPack]()
    pl.set_reader(PlainTextReader())
    pl.add(NLTKSentenceSegmenter())
    pl.add(NLTKWordTokenizer())
    pl.add(NLTKPOSTagger())
    pl.add(CoNLLNERPredictor(), config=config.NER)
    pl.add(SRLPredictor(), config=config.SRL)

    pl.initialize()

    for pack in pl.process_dataset(dataset_dir):
        print(colored("Document", 'red'), pack.pack_name)
        for sentence in pack.get(Sentence):
            sent_text = sentence.text
            print(colored("Sentence:", 'red'), sent_text, "\n")
            # first method to get entry in a sentence
            tokens = [(token.text, token.pos) for token in
                      pack.get(Token, sentence)]
            entities = [(entity.text, entity.ner_type) for entity in
                        pack.get(EntityMention, sentence)]
            print(colored("Tokens:", 'red'), tokens, "\n")
            print(colored("EntityMentions:", 'red'), entities, "\n")

            # second method to get entry in a sentence
            print(colored("Semantic role labels:", 'red'))
            for link in pack.get(PredicateLink, sentence):
                parent: PredicateMention = link.get_parent()  # type: ignore
                child: PredicateArgument = link.get_child()  # type: ignore
                print(f"  - \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")
                entities = [entity.text for entity
                            in pack.get(EntityMention, child)]
                print("      Entities in predicate argument:", entities, "\n")
            print()

            input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_dir)
