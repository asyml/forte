import dill

from nlp.pipeline.data.readers.conll03_reader import CoNLL03Reader
from nlp.pipeline.processors.impl.ner_predictor import (
    CoNLLNERPredictor,
)

reader = CoNLL03Reader(lazy=False)

output_file = 'predict_output.txt'

resource = dill.load(open('resources.pkl', 'rb'))

ner_predictor = CoNLLNERPredictor()
ner_predictor.initialize(resource)
ner_predictor.set_mode(overwrite=True)
# ner_predictor.load_model_checkpoint()

for pack in reader.dataset_iterator(resource.resources[
                                        'config_data'].test_path):
    ner_predictor.process(pack)

    for pred_sentence in pack.get_data(
            context_type="sentence",
            annotation_types={
                "Token": {
                    "fields": ["ner_tag"],
                },
                "Sentence": [],  # span by default
                "EntityMention": {
                }
            }):
        print(pred_sentence)
