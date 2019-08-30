import os
from texar.torch import HParams
from termcolor import colored
import forte.data.ontology.base_ontology as base_ontology
import forte.data.ontology.stanfordnlp_ontology as stanfordnlp_ontology
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors import (
    NLTKPOSTagger, NLTKSentenceSegmenter, NLTKWordTokenizer,
    CoNLLNERPredictor, SRLPredictor, StandfordNLPProcessor)


def StringProcessorExample():
    pl = Pipeline()
    pl.set_reader(StringReader())
    pl.add_processor(NLTKSentenceSegmenter())
    pl.add_processor(NLTKWordTokenizer())
    pl.add_processor(NLTKPOSTagger())

    ner_configs = HParams(
        {
            'storage_path': './NER/resources.pkl',
        },
        CoNLLNERPredictor.default_hparams())

    ner_predictor = CoNLLNERPredictor()

    pl.add_processor(ner_predictor, ner_configs)

    srl_configs = HParams(
        {
            'storage_path': './SRL_model/',
        },
        SRLPredictor.default_hparams()
    )
    pl.add_processor(SRLPredictor(), srl_configs)

    pl.initialize_processors()

    text = (
        "The plain green Norway spruce is displayed in the gallery's foyer. "
        "Wentworth worked as an assistant to sculptor Henry Moore in the "
        "late 1960s. His reputation as a sculptor grew in the 1980s.")

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
            parent = link.get_parent()
            child = link.get_child()
            print(f"  - \"{child.text}\" is role {link.arg_type} of "
                  f"predicate \"{parent.text}\"")
            entities = [entity.text for entity
                        in pack.get(base_ontology.EntityMention, child)]
            print("      Entities in predicate argument:", entities, "\n")
        print()

        input(colored("Press ENTER to continue...\n", 'green'))


def StanfordNLPExample1():
    pl = Pipeline()
    pl.set_reader(StringReader())

    models_path = os.getcwd()
    config = {
        'processors': 'tokenize,pos,lemma,depparse',
        'lang': 'fr',  # Language code for the language to build the Pipeline
        'use_gpu': False
    }
    pl.add_processor(processor=StandfordNLPProcessor(models_path),
                     config=config)
    pl.set_ontology(stanfordnlp_ontology)

    pl.initialize_processors()

    text = (
        "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie."
    )

    pack = pl.process(text)
    for sentence in pack.get(stanfordnlp_ontology.Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        tokens = [(token.text, token.pos_tag, token.lemma) for token in
                  pack.get(stanfordnlp_ontology.Token, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")

        print(colored("Dependecy Relations:", 'red'))
        for link in pack.get(
                stanfordnlp_ontology.Dependency, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(colored(child.text, 'cyan'),
                  "has relation",
                  colored(link.rel_type, 'green'),
                  "of parent",
                  colored(parent.text, 'cyan'))

        print("\n----------------------\n")


def StanfordNLPExample2():
    pl = Pipeline()
    pl.set_reader(StringReader())
    config = {
        'processors': 'tokenize,pos,lemma,depparse',
        'lang': 'en',  # Language code for the language to build the Pipeline
        'use_gpu': False
    }
    models_path = os.getcwd()
    pl.add_processor(processor=StandfordNLPProcessor(models_path),
                     config=config)
    pl.set_ontology(stanfordnlp_ontology)
    pl.initialize_processors()

    text = (
        "The plain green Norway spruce is displayed in the gallery's foyer. "
         "Wentworth worked as an assistant to sculptor Henry Moore in the "
         "late 1960s. His reputation as a sculptor grew in the 1980s.")

    pack = pl.process(text)

    for sentence in pack.get(stanfordnlp_ontology.Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        tokens = [(token.text, token.pos_tag, token.lemma) for token in
                  pack.get(stanfordnlp_ontology.Token, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")

        print(colored("Dependecy Relations:", 'red'))
        for link in pack.get(
                stanfordnlp_ontology.Dependency, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(colored(child.text, 'cyan'),
                  "has relation",
                  colored(link.rel_type, 'green'),
                  "of parent",
                  colored(parent.text, 'cyan'))

        print("\n----------------------\n")


if __name__ == '__main__':
    StanfordNLPExample1()
    StanfordNLPExample2()
    StringProcessorExample()
