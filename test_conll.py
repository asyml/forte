from nlp.pipeline.io.readers.conll03_reader import Conll03Reader

reader = Conll03Reader(lazy=False)

train_path = "data/conll03_english/eng.train.bioes.conll"
pack = reader.read(train_path)

print(pack.internal_metas["Sentence"].fields_created)

lists = pack.get_data(
    context_type="sentence",
    annotation_types={
        "Token": ["chunk_tag", "pos_tag", "ner_tag"],
        "Sentence": [] # span by default
    },
)
print(next(lists)['Token']['text'])
