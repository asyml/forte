import argparse
import json
import torch
import yaml
from forte.data import base_ontology

from forte.data import PlainSentenceTxtgenReader
from nlp.forte.multipack_pipeline import MultiPackPipeline


pl = MultiPackPipeline()
reader = PlainSentenceTxtgenReader(lazy=False)
pl.set_reader(reader)

pl.init_from_config_path("sample_multipack_pipeline_gpt.yml")

print("\nFinished loading\n")

input_dir = "data/en_vi/test/src/test.en"
input_pack_name = "input_src"
output_pack_name = "output_tgt"

multipack = pl.process_one(input_dir)

src_cnt = len(list(multipack.packs[input_pack_name].get(
    base_ontology.Sentence)))
tgt_cnt = len(list(multipack.packs[output_pack_name].get(
    base_ontology.Sentence)))
link_cnt = len(multipack.links)
print(f'sentence_cnt: src{src_cnt}, tgt{tgt_cnt}, link_cnt{link_cnt}')

with open("multipack_output.txt", "w+") as fout:
    parsed = json.loads(reader.serialize_instance(multipack))
    fout.write(json.dumps(parsed, indent=4))
