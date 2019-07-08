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
ner_predictor.load_model_checkpoint()

opened_file = open(output_file, "w+")

for pack in reader.dataset_iterator(resource.resources[
                                        'config_data'].test_path):
    ner_predictor.process(pack)
    for pred_sentence, tgt_sentence in zip(
            pack.get_data(
                context_type="sentence",
                annotation_types={
                    "Token": {
                        "component": "ner_predictor",
                        "fields": ["chunk_tag", "pos_tag", "ner_tag"],
                    },
                    "Sentence": [],  # span by default
                },
            ),
            pack.get_data(
                context_type="sentence",
                annotation_types={
                    "Token": {"fields": ["chunk_tag", "pos_tag", "ner_tag"]},
                    "Sentence": [],  # span by default
                },
            ),
    ):

        pred_tokens, tgt_tokens = (
            pred_sentence["Token"],
            tgt_sentence["Token"],
        )
        for i in range(len(pred_tokens["text"])):
            w = pred_tokens["text"][i]
            p = pred_tokens["pos_tag"][i]
            ch = pred_tokens["chunk_tag"][i]
            tgt = tgt_tokens["ner_tag"][i]
            pred = pred_tokens["ner_tag"][i]

            opened_file.write(
                "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred)
            )

        opened_file.write("\n")
opened_file.close()


