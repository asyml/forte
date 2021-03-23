import os

from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.data.readers import RawDataDeserializeReader
from forte.processors.writers import PackNameJsonPackWriter
from forte.processors.allennlp_processors import AllenNLPProcessor
from forte.processors.spacy_processors import SpacyProcessor

input_artifact_dir = '/Users/jenny.zhang/Documents/materials/'
output_artifact_dir = '/Users/jenny.zhang/Documents/materials/'


plp = Pipeline[DataPack]()
plp.enforce_consistency()
plp.set_reader(RawDataDeserializeReader())
# Using SpacyProcessor to segment the sentences
plp.add(SpacyProcessor(), config={
    'processors': '',
    'lang': "en_core_web_sm",  # Language code to build the Pipeline
    'use_gpu': False
})
plp.add(AllenNLPProcessor(), config={
    'processors': "tokenize",
    'tag_formalism': "stanford",
    'overwrite_entries': False,
    'allow_parallel_entries': True
})
plp.add(
    PackNameJsonPackWriter(),
    {
        'output_dir': output_artifact_dir,
        'indent': 2,
        'overwrite': True,
    }
)
plp.initialize()
with open(os.path.join(input_artifact_dir, "ner_input_eng_datapack.json"), "r") as f:
    lines = f.read()
    print(f"reading content:  {lines}")
    plp.process([lines])
