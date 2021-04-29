# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from termcolor import colored

from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence


def create_pipeline(config_path: str) -> Pipeline[MultiPack]:
    pl = Pipeline[MultiPack]()
    pl.init_from_config_path(config_path)
    print("\nFinished loading\n")
    return pl


if __name__ == '__main__':
    # This line adds a reader and 3 processors in to the pipeline
    # 1. forte.data.readers.MultiPackSentenceReader
    # 2. forte.processors.third_party.TextGenerationProcessor
    # 3. forte.processors.third_party.NLTKWordTokenizer
    # 4. forte.processors.third_party.NLTKPOSTagger
    nlp = create_pipeline('sample_multipack_pipeline_gpt.yml')
    nlp.initialize()

    input_dir = "data/"

    multipack: MultiPack = nlp.process_one(input_dir)

    input_pack_name = "input_src"
    output_pack_name = "output_tgt"

    src_cnt = len(list(multipack.get_pack(input_pack_name).get(Sentence)))
    tgt_cnt = len(list(multipack.get_pack(output_pack_name).get(Sentence)))
    link_cnt = multipack.num_links
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
