import argparse
import yaml
from termcolor import colored
from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import PlainTextReader
from forte.processors import (
    CoNLLNERPredictor, SRLPredictor)
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter

from ft.onto.base_ontology import Token, Sentence, EntityMention, \
    PredicateLink, PredicateArgument, PredicateMention


parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, default="data/",
                    help="Data directory to read the text files from")


def main(dataset_dir: str):
    config = yaml.safe_load(open("config.yml", "r"))
    config = HParams(config, default_hparams=None)

    pl = Pipeline()
    pl.set_reader(PlainTextReader())
    pl.add_processor(NLTKSentenceSegmenter())
    pl.add_processor(NLTKWordTokenizer())
    pl.add_processor(NLTKPOSTagger())
    pl.add_processor(CoNLLNERPredictor(), config=config.NER)
    pl.add_processor(SRLPredictor(), config=config.SRL)

    pl.initialize()

    for pack in pl.process_dataset(dataset_dir):
        print(colored("Document", 'red'), pack.meta.doc_id)
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
