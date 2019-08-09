import argparse
import os
import json
import texar.torch as tx
import torch
import yaml
from termcolor import colored
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.ontology.base_ontology import (
    Token, Sentence, EntityMention, PredicateLink)

from nlp.pipeline.data.readers import PlainSentenceTxtgenReader
from nlp.pipeline.multipack_pipeline import MultiPackPipeline


pl = MultiPackPipeline()
reader = PlainSentenceTxtgenReader(lazy=False)
pl.set_reader(reader)

pl.init_from_config_path("sample_multipack_pipeline_gpt.yml")

print("\nFinished loading\n")

input_dir = "data/en_vi/test/src/test.en"
input_pack_name = "input_src"
output_pack_name = "output_tgt"

multipack = pl.process(input_dir)

src_cnt = len(list(multipack.packs[input_pack_name].get(
    base_ontology.Sentence)))
tgt_cnt = len(list(multipack.packs[output_pack_name].get(
    base_ontology.Sentence)))
link_cnt = len(multipack.links)
print(f'sentence_cnt: src{src_cnt}, tgt{tgt_cnt}, link_cnt{link_cnt}')

with open("multipack_output.txt", "w+") as fout:
    parsed = json.loads(reader.serialize_instance(multipack))
    fout.write(json.dumps(parsed, indent=4))
