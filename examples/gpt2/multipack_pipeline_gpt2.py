import json

from termcolor import colored

from forte.multipack_pipeline import MultiPackPipeline

from ft.onto.base_ontology import Token, Sentence

pl = MultiPackPipeline()

pl.init_from_config_path("sample_multipack_pipeline_gpt.yml")

print("\nFinished loading\n")

input_pack_name = "input_src"
output_pack_name = "output_tgt"

input_dir = "data/"

multipack = pl.process_one(input_dir)

src_cnt = len(list(multipack.get_pack(input_pack_name).get(Sentence)))
tgt_cnt = len(list(multipack.get_pack(output_pack_name).get(Sentence)))
link_cnt = len(multipack.links)
print(f'sentence_cnt: src{src_cnt}, tgt{tgt_cnt}, link_cnt{link_cnt}')

with open("multipack_output.txt", "w+") as fout:
    input_pack = multipack.get_pack(input_pack_name)
    output_pack = multipack.get_pack(output_pack_name)
    for context, gen_sentence in zip(input_pack.get(Sentence),
                        output_pack.get(Sentence)):
        print(colored("Initial Context", "green"), context.text)
        print(colored("Generated Sentence", "green"), gen_sentence.text)
        print("======================TAGS======================")
        for token in output_pack.get(entry_type=Token,
                                     range_annotation=gen_sentence):
            print(colored("Token", "red"), token.text,
                  colored("POS Tag", "red"), token.pos)
        print("======================END======================")
    parsed = json.loads(multipack.serialize())
    fout.write(json.dumps(parsed, indent=4))
