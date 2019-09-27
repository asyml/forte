import os

from termcolor import colored
from texar.torch import HParams

import forte.data.ontology.base_ontology as base_ontology
import forte.data.ontology.stanfordnlp_ontology as stanford
from forte.data.ontology.ontonotes_ontology import (
    PredicateMention, PredicateArgument)
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors import (
    NLTKPOSTagger, NLTKSentenceSegmenter, NLTKWordTokenizer,
    CoNLLNERPredictor, SRLPredictor)
from forte.processors.StanfordNLP_processor import StandfordNLPProcessor
from forte.processors.simple_writers import SimpleJsonPackWriter


def string_processor_example(ner_model_dir: str, srl_model_dir: str):
    pl = Pipeline()
    pl.set_reader(StringReader())
    pl.add_processor(NLTKSentenceSegmenter())
    pl.add_processor(NLTKWordTokenizer())
    pl.add_processor(NLTKPOSTagger())

    """ner_configs = HParams(
        {
            'storage_path': os.path.join(ner_model_dir, 'resources.pkl')
        },
        CoNLLNERPredictor.default_hparams())

    ner_predictor = CoNLLNERPredictor()

    pl.add_processor(ner_predictor, ner_configs)"""

    srl_configs = HParams(
        {
            'storage_path': srl_model_dir,
        },
        SRLPredictor.default_hparams()
    )
    pl.add_processor(SRLPredictor(), srl_configs)

    pl.initialize_processors()

    text = ["The plain green Norway spruce is displayed in the gallery's foyer. "
        "Wentworth worked as an assistant to sculptor Henry Moore in the "
        "late 1960s. His reputation as a sculptor grew in the 1980s."]

    pack = pl.process_one(text)

    for sentence in pack.get(base_ontology.Sentence):
        sent_text = sentence.text
        print(colored("base_ontology.Sentence:", 'red'), sent_text, "\n")
        # first method to get entry in a sentence
        tokens = [(token.text, token.pos_tag) for token in
                  pack.get(base_ontology.Token, sentence)]
        entities = [(entity.text, entity.ner_type) for entity in
                    pack.get(base_ontology.EntityMention, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")
        print(colored("EntityMentions:", 'red'), entities, "\n")

        # second method to get entry in a sentence
        print(colored("Semantic role labels:", 'red'))
        for link in pack.get(
                base_ontology.PredicateLink, sentence):
            parent: PredicateMention = link.get_parent()  # type: ignore
            child: PredicateArgument = link.get_child()  # type: ignore
            print(f"  - \"{child.text}\" is role {link.arg_type} of "
                  f"predicate \"{parent.text}\"")
            entities = [entity.text for entity
                        in pack.get(base_ontology.EntityMention, child)]
            print("      Entities in predicate argument:", entities, "\n")
        print()

        input(colored("Press ENTER to continue...\n", 'green'))


def stanford_nlp_example1(lang: str, text: str, output_config: HParams):
    pl = Pipeline()
    pl.set_reader(StringReader())

    models_path = os.getcwd()
    config = HParams(
        {
            'processors': 'tokenize,pos,lemma,depparse',
            'lang': lang,
            # Language code for the language to build the Pipeline
            'use_gpu': False
        },
        StandfordNLPProcessor.default_hparams()
    )
    pl.add_processor(processor=StandfordNLPProcessor(models_path),
                     config=config)
    pl.add_processor(processor=SimpleJsonPackWriter(),
                     config=output_config)
    pl.set_ontology(stanford)

    pl.initialize_processors()

    pack = pl.process(text)
    for sentence in pack.get(stanford.Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        tokens = [(token.text, token.pos_tag, token.lemma) for token in
                  pack.get(stanford.Token, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")

        print(colored("Dependency Relations:", 'red'))
        for link in pack.get(
                stanford.Dependency, sentence):
            parent: stanford.Token = link.get_parent()  # type: ignore
            child: stanford.Token = link.get_child()  # type: ignore
            print(colored(child.text, 'cyan'),
                  "has relation",
                  colored(link.rel_type, 'green'),
                  "of parent",
                  colored(parent.text, 'cyan'))

        print("\n----------------------\n")


def main():
    import sys
    ner_dir, srl_dir = sys.argv[  # pylint: disable=unbalanced-tuple-unpacking
                       1:3]

    output_config = HParams(
        {
            'output_dir': '.'
        },
        SimpleJsonPackWriter.default_hparams(),
    )

    eng_text = "The plain green Norway spruce is displayed in the gallery's " \
               "foyer. Wentworth worked as an assistant to sculptor Henry " \
               "Moore in the late 1960s. His reputation as a sculptor grew " \
               "in the 1980s."

    fr_text = "Van Gogh grandit au sein d'une famille de " \
              "l'ancienne bourgeoisie."

    stanford_nlp_example1('en', eng_text, output_config)
    stanford_nlp_example1('fr', fr_text, output_config)

    print("Running SRL now...")
    string_processor_example(ner_dir, srl_dir)


if __name__ == '__main__':
    main()
