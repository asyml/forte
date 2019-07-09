import dill
from termcolor import colored

from nlp.pipeline.data import CoNLL03Ontology as Ont
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors import (NLTKPOSTagger, NLTKSentenceSegmenter,
                                     NLTKWordTokenizer)
from nlp.pipeline.processors.impl import CoNLLNERPredictor, SRLPredictor


def main():


    pl = Pipeline()
    pl.processors.append(NLTKSentenceSegmenter())
    pl.processors.append(NLTKWordTokenizer())
    pl.processors.append(NLTKPOSTagger())

    ner_resource = dill.load(open('./NER/resources.pkl', 'rb'))
    ner_predictor = CoNLLNERPredictor()
    ner_predictor.initialize(ner_resource)
    ner_predictor.load_model_checkpoint()
    pl.processors.append(ner_predictor)

    pl.processors.append(SRLPredictor(model_dir="./SRL_model/"))

    text = "The plain green Norway spruce is displayed in the gallery's foyer. " \
           "Wentworth worked as an assistant to sculptor Henry Moore in the " \
           "late 1960s. His reputation as a sculptor grew in the 1980s." \

    pack = pl.process(text)

    for sentence in pack.get(Ont.Sentence):
        sent_text = sentence.text
        print(colored("Sentence:",'red'), sent_text, "\n")
        # first method to get entry in a sentence
        print(colored("Semantic role labels:",'red'))
        for link in pack.get(
            Ont.PredicateLink, sentence,
                component=pl.processors[-1].component_name):
            parent = link.get_parent()
            child = link.get_child()
            print(f"  - \"{child.text}\" is role {link.arg_type} of "
                  f"predicate \"{parent.text}\"")
        print()
        # second method to get entry in a sentence
        tokens = [(token.text, token.pos_tag) for token in
                  pack.get(Ont.Token, sentence)]
        entities = [(entity.text, entity.ner_type) for entity in
                    pack.get(Ont.EntityMention, sentence)]
        print(colored("Tokens:",'red'), tokens, "\n")
        print(colored("EntityMention:",'red'), entities, "\n")

        input(colored("Press ENTER to continue...\n",'green'))


if __name__ == '__main__':
    main()
