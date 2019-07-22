import pickle
from termcolor import colored

from nlp.pipeline.data.ontology.conll03_ontology import (
    Token, Sentence, EntityMention, PredicateLink)
from nlp.pipeline.pipeline import Pipeline

from nlp.pipeline.processors.impl import (
    NLTKPOSTagger, NLTKSentenceSegmenter, NLTKWordTokenizer,
    CoNLLNERPredictor, SRLPredictor)


def main():

    pl = Pipeline()
    pl.initialize(ontology="conll03_ontology")
    pl.processors.append(NLTKSentenceSegmenter())
    pl.processors.append(NLTKWordTokenizer())
    pl.processors.append(NLTKPOSTagger())

    ner_resource = pickle.load(open('./NER/resources.pkl', 'rb'))
    ner_predictor = CoNLLNERPredictor()
    ner_predictor.initialize(ner_resource)
    ner_predictor.load_model_checkpoint()
    pl.processors.append(ner_predictor)

    pl.processors.append(SRLPredictor(model_dir="./SRL_model/"))

    text = (
        "The plain green Norway spruce is displayed in the gallery's foyer. "
        "Wentworth worked as an assistant to sculptor Henry Moore in the "
        "late 1960s. His reputation as a sculptor grew in the 1980s.")

    pack = pl.process(text)

    for sentence in pack.get(Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
        # first method to get entry in a sentence
        tokens = [(token.text, token.pos_tag) for token in
                  pack.get(Token, sentence)]
        entities = [(entity.text, entity.ner_type) for entity in
                    pack.get(EntityMention, sentence)]
        print(colored("Tokens:", 'red'), tokens, "\n")
        print(colored("EntityMentions:", 'red'), entities, "\n")

        # second method to get entry in a sentence
        print(colored("Semantic role labels:", 'red'))
        for link in pack.get(
                PredicateLink, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(f"  - \"{child.text}\" is role {link.arg_type} of "
                  f"predicate \"{parent.text}\"")
            entities = [entity.text for entity
                        in pack.get(EntityMention, child)]
            print("      Entities in predicate argument:", entities, "\n")
        print()

        input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == '__main__':
    main()
