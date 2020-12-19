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

import logging
from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.base.data_selector_for_da import RandomDataSelectorForDA
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from forte.data.selector import AllPackSelector
from forte.data.caster import MultiPackBoxer
from forte.processors.base.data_augment_processor \
    import ReplacementDataAugmentProcessor


logging.root.setLevel(logging.INFO)


def main():
    indexer_name = "da_selector"
    size = 2
    da_selector_config = {"index_config": {"index_name": indexer_name},
                          "size": size}
    boxer_config = {
        'pack_name': 'input'
    }
    da_processor_config = {
        'augment_entry': "ft.onto.base_ontology.Token",
        'other_entry_policy': {
            'type': '',
            'kwargs': {
                "ft.onto.base_ontology.Document": "auto_align",
                "ft.onto.base_ontology.Sentence": "auto_align"
            }
        },
        'type': 'data_augmentation_op',
        'data_aug_op': 'forte.processors.data_augment.algorithms.'
                       'dictionary_replacement_op.DictionaryReplacementOp',
        'data_aug_op_config': {
            'type': '',
            'kwargs': {
                "dictionary_class": "forte.processors.data_augment.algorithms."
                                    "dictionary.WordnetDictionary",
                "prob": 1.0,
                "lang": "eng",
            }
        },
        'augment_pack_names': {
            'kwargs': {
                'input': 'augmented_input'
            }
        }
    }

    nlp: Pipeline[MultiPack] = Pipeline()
    nlp.set_reader(RandomDataSelectorForDA(), config=da_selector_config)
    nlp.add(component=MultiPackBoxer(), config=boxer_config)
    nlp.add(component=NLTKWordTokenizer(), selector=AllPackSelector())
    nlp.add(component=NLTKPOSTagger(), selector=AllPackSelector())
    nlp.add(component=ReplacementDataAugmentProcessor(),
            config=da_processor_config)

    nlp.initialize()

    for _, m_pack in enumerate(nlp.process_dataset()):
        aug_pack = m_pack.get_pack('augmented_input')
        logging.info(aug_pack.text)


if __name__ == "__main__":
    main()
