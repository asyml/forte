import pickle
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers.conll03_reader import CoNLL03Reader
from nlp.pipeline.processors.impl.ner_predictor import (
    CoNLLNERPredictor,
)
from nlp.pipeline.common.resources import Resources


reader = CoNLL03Reader(lazy=False)

output_file = 'predict_output.txt'

resources: Resources = Resources()
resources.load('resources.pkl')

ner_predictor = CoNLLNERPredictor()
ner_predictor.initialize(resources)
ner_predictor.set_mode(overwrite=True)

for pack in reader.dataset_iterator(resources.resources[
                                        'config_data'].test_path):
    ner_predictor.process(pack)

    for pred_sentence in pack.get_data(
            context_type="sentence",
            requests={
                base_ontology.Token: {
                    "fields": ["ner_tag"],
                },
                base_ontology.Sentence: [],  # span by default
                base_ontology.EntityMention: {
                }
            }):
        print(pred_sentence)
