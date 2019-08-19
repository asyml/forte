from forte.data.ontology import base_ontology
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import (
    CoNLLNERPredictor,
)
from forte.common.resources import Resources


reader = CoNLL03Reader(lazy=False)

output_file = 'predict_output.txt'

resources: Resources = Resources()
resources.load('resources.pkl')

ner_predictor = CoNLLNERPredictor()
ner_predictor.initialize(resources)

for pack in reader.iter(resources.resources[
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
