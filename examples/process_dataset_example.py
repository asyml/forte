import dill
from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.data import CoNLL03Ontology as Ont
from nlp.pipeline.processors.impl import CoNLLNERPredictor
from nlp.pipeline.processors import (
    NLTKSentenceSegmenter, NLTKWordTokenizer,
    NLTKPOSTagger
)


dataset = {
    "dataset_dir": "./bbc",
    "dataset_format": "plain"
}

pl = Pipeline()
pl.processors.append(NLTKSentenceSegmenter())
pl.processors.append(NLTKWordTokenizer())
pl.processors.append(NLTKPOSTagger())

ner_resource = dill.load(open('./ner/resources.pkl', 'rb'))
ner_predictor = CoNLLNERPredictor()
ner_predictor.initialize(ner_resource)
ner_predictor.load_model_checkpoint()
pl.processors.append(ner_predictor)


for pack in pl.process_dataset(dataset):
    print(pack.meta.doc_id)
    for sentence in pack.get(Ont.Sentence):
        sent_text = sentence.text
        print("Sentence:", sent_text)
        # first method to get entry in a sentence
        for link in pack.get(Ont.PredicateLink, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(f"Links: {parent.text} is {link.rel_type} {child.text}")

        # second method to get entry in a sentence
        tokens = [(token.text, token.pos_tag) for token in
                  pack.get(Ont.Token, sentence)]
        print("Tokens:", tokens)

        input()

