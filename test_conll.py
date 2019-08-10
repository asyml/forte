from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.ontology.base_ontology import Token, Sentence
import importlib

reader = CoNLL03Reader(lazy=False)

config_data = importlib.import_module("config_data")
train_path = config_data.train_path  # type: ignore
pack = reader.iter('examples/NER/' + train_path)

pack = list(pack)[0]

print(pack.internal_metas["Sentence"].fields_created)

lists = pack.get_data(
    context_type="sentence",
    requests={
        Token: ["chunk_tag", "pos_tag", "ner_tag"],
        Sentence: []  # span by default
    },
)
print(next(lists)['Token'])
