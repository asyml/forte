from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors.dummy_processor import RelationOntology as Ont
from nlp.pipeline.processors.sentence_predictor import NLTKSentenceSegmenter
from nlp.pipeline.processors.tokenization_predictor import NLTKWordTokenizer
from nlp.pipeline.processors.postag_predictor import NLTKPOSTagger

kwargs = {
    "dataset": {
        "dataset_dir": "./bbc",
        "dataset_format": "plain"
    }
}

pl = Pipeline(**kwargs)
pl.processors.append(NLTKSentenceSegmenter())
pl.processors.append(NLTKWordTokenizer())
pl.processors.append(NLTKPOSTagger())
# pl.processors.append(ner)
# pl.processors.append(srl)


for pack in pl.run():
    print(pack.meta.doc_id)
    print(pack.text)
    for sentence in pack.get(Ont.Sentence):
        sent_text = sentence.text
        print(sent_text)
        # first method to get entry in a sentence
        for link in pack.get(Ont.RelationLink, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(f"Relation: {parent.text} is {link.rel_type} {child.text}")

        # second method to get entry in a sentence
        tokens = [(token.text, token.pos_tag) for token in pack.get(Ont.Token, sentence)]
        print("Tokens:",tokens)
    input()
