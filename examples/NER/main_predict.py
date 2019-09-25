from forte import Pipeline
from forte.data.ontology import base_ontology
from forte.data.ontology.conll03_ontology import Sentence
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import (
    CoNLLNERPredictor,
)
from forte.common.resources import Resources

resources: Resources = Resources()
resources.load('resources.pkl')

pl = Pipeline(resources)
pl.set_reader(CoNLL03Reader(lazy=False))
pl.add_processor(CoNLLNERPredictor())

for pack in pl.process_dataset('ner_data/conll03_english/test'):
    for pred_sentence in pack.get_data(
            context_type=Sentence,
            request={
                base_ontology.Token: {
                    "fields": ["ner_tag"],
                },
                base_ontology.Sentence: [],  # span by default
                base_ontology.EntityMention: {
                }
            }):
        print(pred_sentence)
