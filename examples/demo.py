from nlp.pipeline.pipeline import Pipeline
from nlp.pipeline.processors.dummy_processor import DummyRelationExtractor
from nlp.pipeline.processors.dummy_processor import RelationOntology as Ont

kwargs = {
    "dataset": {
        "dataset_dir": "./ontonotes_sample_dataset",
        "dataset_format": "Ontonotes"
    }
}

pl = Pipeline(**kwargs)
pl.processors.append(DummyRelationExtractor())

for pack in pl.run():
    for sentence in pack.get(Ont.Sentence):
        sent_text = sentence.text

        # first method to get entry in a sentence
        for link in pack.get(Ont.RelationLink, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(f"Relation: {parent.text} is {link.rel_type} {child.text}")

        # second method to get entry in a sentence
        tokens = [token.text for token in pack.get(Ont.Token, sentence)]
        print("Tokens:",tokens)

