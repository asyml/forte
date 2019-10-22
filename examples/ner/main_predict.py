import yaml

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import CoNLLNERPredictor
from ft.onto import base_ontology

config_data = yaml.safe_load(open("config_data.yml", "r"))
config_model = yaml.safe_load(open("config_model.yml", "r"))

config = HParams({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)


pl = Pipeline()
pl.set_reader(CoNLL03Reader())
pl.add_processor(CoNLLNERPredictor(), config=config)

pl.initialize()

for pack in pl.process_dataset(config.config_data.test_path):
    for pred_sentence in pack.get_data(
            context_type=base_ontology.Sentence,
            request={
                base_ontology.Token: {"fields": ["ner"]},
                base_ontology.Sentence: [],  # span by default
                base_ontology.EntityMention: {}
            }):
        print("============================")
        print(pred_sentence["context"])
        print(pred_sentence["Token"]["ner"])
        print("============================")
