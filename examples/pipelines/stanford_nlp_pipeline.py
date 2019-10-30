import os

from termcolor import colored
from texar.torch import HParams

from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.stanfordnlp_processor import StandfordNLPProcessor

from ft.onto.base_ontology import Token, Sentence, Dependency


def stanford_nlp_example(lang: str, text: str):
    pl = Pipeline()
    pl.set_reader(StringReader())

    models_path = os.getcwd()
    config = HParams({
        'processors': 'tokenize,pos,lemma,depparse',
        'lang': lang,
        # Language code for the language to build the Pipeline
        'use_gpu': False
    }, StandfordNLPProcessor.default_hparams())
    pl.add_processor(processor=StandfordNLPProcessor(models_path),
                     config=config)

    pl.initialize()

    pack = pl.process(text)
    for sentence in pack.get(Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        tokens = [(token.text, token.pos_tag, token.lemma) for token in
                  pack.get(Token, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")

        print(colored("Dependency Relations:", 'red'))
        for link in pack.get(Dependency, sentence):
            parent: Token = link.get_parent()  # type: ignore
            child: Token = link.get_child()  # type: ignore
            print(colored(child.text, 'cyan'),
                  "has relation",
                  colored(link.rel_type, 'green'),
                  "of parent",
                  colored(parent.text, 'cyan'))

        print("\n----------------------\n")


def main():

    eng_text = "The plain green Norway spruce is displayed in the gallery's " \
               "foyer. Wentworth worked as an assistant to sculptor Henry " \
               "Moore in the late 1960s. His reputation as a sculptor grew " \
               "in the 1980s."

    fr_text = "Van Gogh grandit au sein d'une famille de " \
              "l'ancienne bourgeoisie."

    stanford_nlp_example('en', eng_text)
    stanford_nlp_example('fr', fr_text)


if __name__ == '__main__':
    main()
