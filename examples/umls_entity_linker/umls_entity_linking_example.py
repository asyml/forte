# Copyright 2020 The Forte Authors. All Rights Reserved.
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

from texar.torch import HParams

from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import LinkedMention

from umls_entity_linker_processor import ScispaCyUMLSEntityLinker


def umls_entity_linker_example(text: str):
    pl = Pipeline()
    pl.set_reader(StringReader())

    config = HParams({
            'model_name': 'en_core_sci_sm',
            'model_version': '0.2.4',
            'resolve_abbreviations': False,
            'overwrite_entries': False,
            'allow_parallel_entries': True
        }, ScispaCyUMLSEntityLinker.default_configs())
    pl.add_processor(processor=ScispaCyUMLSEntityLinker(),
                     config=config)

    pl.initialize()
    pack = pl.process(text)

    for linked_entities in pack.get(LinkedMention):
        print(f'Entity {linked_entities.text} is linked to entity ids '
              f'{list(linked_entities.linked_kb_ids.keys())} '
              f'in {linked_entities.kb} Knowledge Base')

        print("\n----------------------\n")


def main():
    eng_text = "Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily."

    umls_entity_linker_example(eng_text)


if __name__ == '__main__':
    main()
