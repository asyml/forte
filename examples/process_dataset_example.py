import dill

from nlp.pipeline.data import CoNLL03Ontology as Ont
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors import (NLTKPOSTagger, NLTKSentenceSegmenter,
                                     NLTKWordTokenizer)
from nlp.pipeline.processors.impl import CoNLLNERPredictor, SRLPredictor


def main():
    dataset = {
        "dataset_dir": "../../bbc",
        "dataset_format": "plain"
    }
    #dataset = {
    #    "dataset_dir": "./ontonotes_sample_dataset",
    #    "dataset_format": "ontonotes"
    #}

    pl = Pipeline()
    pl.processors.append(NLTKSentenceSegmenter())
    pl.processors.append(NLTKWordTokenizer())
    pl.processors.append(NLTKPOSTagger())

    ner_resource = dill.load(open('./NER/resources.pkl', 'rb'))
    ner_predictor = CoNLLNERPredictor()
    ner_predictor.set_mode(True)
    ner_predictor.initialize(ner_resource)
    ner_predictor.load_model_checkpoint()
    pl.processors.append(ner_predictor)
    pl.processors.append(SRLPredictor(model_dir="./SRL_model/"))

    for pack in pl.process_dataset(dataset):
        print(pack.meta.doc_id)
        for sentence in pack.get(Ont.Sentence):
            sent_text = sentence.text
            print("Sentence:", sent_text)
            # first method to get entry in a sentence
            for link in pack.get(
                    Ont.PredicateLink, sentence,
                    component=pl.processors[-1].component_name):
                parent = link.get_parent()
                child = link.get_child()
                print(f"SRL: \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")

            # second method to get entry in a sentence
            tokens = [(token.text, token.pos_tag) for token in
                      pack.get(Ont.Token, sentence)]
            entities = [entity.text for entity in pack.get(Ont.EntityMention, sentence)]
            print("Tokens:", tokens, "EntityMention", entities)

            input("Press ENTER to continue...")


if __name__ == '__main__':
    main()
