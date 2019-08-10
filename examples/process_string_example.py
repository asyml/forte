from termcolor import colored
from texar.torch import HParams

from forte.data.ontology.base_ontology import (
    Token, Sentence, EntityMention, PredicateLink)
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors import (
    NLTKPOSTagger, NLTKSentenceSegmenter, NLTKWordTokenizer,
    CoNLLNERPredictor, SRLPredictor)


def main():

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
