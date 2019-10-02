import yaml
from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.ontology import base_ontology
from forte.data.ontology.conll03_ontology import Sentence
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import (
    CoNLLNERPredictor,
)
from forte.common.resources import Resources

resources: Resources = Resources()
resources.load('resources.pkl')

config_data = yaml.safe_load(open("config_data.yml", "r"))
config_model = yaml.safe_load(open("config_model.yml", "r"))

config = HParams({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)

pl = Pipeline(resources)
pl.set_reader(CoNLL03Reader())
pl.add_processor(CoNLLNERPredictor(), config=config)

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
