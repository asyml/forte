import os
import sys
import dill
from termcolor import colored

from nlp.pipeline.data.readers import CoNLL03Ontology as Ont
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors.impl import (NLTKPOSTagger, NLTKSentenceSegmenter,
                                          NLTKWordTokenizer, CoNLLNERPredictor,
                                          SRLPredictor)



def main(dataset_dir, ner_model_path, srl_model_path):
    dataset = {
        "dataset_dir": dataset_dir,
        "dataset_format": "plain"
    }

    pl = Pipeline()
    pl.initialize(ontology="CoNLL03Ontology")
    pl.add_processor(NLTKSentenceSegmenter())
    pl.add_processor(NLTKWordTokenizer())
    pl.add_processor(NLTKPOSTagger())

    ner_resource = dill.load(
        open(os.path.join(ner_model_path, 'resources.pkl'), 'rb'))
    ner_predictor = CoNLLNERPredictor()
    ner_predictor.initialize(ner_resource)

    pl.add_processor(ner_predictor)

    pl.add_processor(SRLPredictor(model_dir=srl_model_path))

    for pack in pl.process_dataset(dataset):
        print(colored("Document", 'red'), pack.meta.doc_id)
        for sentence in pack.get(Ont.Sentence):
            sent_text = sentence.text
            print(colored("Sentence:", 'red'), sent_text, "\n")
            # first method to get entry in a sentence
            tokens = [(token.text, token.pos_tag) for token in
                      pack.get(Ont.Token, sentence)]
            entities = [(entity.text, entity.ner_type) for entity in
                        pack.get(Ont.EntityMention, sentence)]
            print(colored("Tokens:", 'red'), tokens, "\n")
            print(colored("EntityMentions:", 'red'), entities, "\n")

            # second method to get entry in a sentence
            print(colored("Semantic role labels:", 'red'))
            for link in pack.get(
                    Ont.PredicateLink, sentence):
                parent = link.get_parent()
                child = link.get_child()
                print(f"  - \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")
                entities = [entity.text for entity
                            in pack.get(Ont.EntityMention, child)]
                print("      Entities in predicate argument:", entities, "\n")
            print()

            input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == '__main__':
    data_dir, ner_dir, srl_dir = sys.argv[1:]
    main(data_dir, ner_dir, srl_dir)
