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

import os
from typing import Iterable, Dict, Tuple

from forte.data.selector import AllPackSelector
from forte.data.caster import MultiPackBoxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import MultiPackLink
from forte.data.readers import LargeMovieReader
from forte.models.imdb_text_classifier.model import IMDBClassifier
from forte.pipeline import Pipeline
from forte.processors.base import ReplacementDataAugmentProcessor
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp
from forte.processors.nltk_processors import NLTKWordTokenizer, NLTKPOSTagger
from ft.onto.base_ontology import Document, Sentence, Token

import config_data
import config_classifier


class TmpReplacementDataAugmentProcessor(ReplacementDataAugmentProcessor):
    def new_pack(self):
        return MultiPack()


def main(argv=None):
    pipeline = Pipeline[MultiPack]()
    reader = LargeMovieReader()
    pipeline.set_reader(reader)
    pipeline.add(component=NLTKWordTokenizer())
    pipeline.add(component=NLTKPOSTagger())
    pipeline.add(MultiPackBoxer())

    model_class_name = (
        "forte.processors.data_augment.algorithms."
        "machine_translator.MarianMachineTranslator"
    )

    processor_config = {
        'augment_entry': "ft.onto.base_ontology.Token",
        'other_entry_policy': {
            "entry": [
                "ft.onto.base_ontology.Document",
                "ft.onto.base_ontology.Sentence"
            ],
            "policy": ["auto_align", "auto_align"],
        },
        "kwargs": {
            'data_aug_op': "forte.processors.data_augment.algorithms.back_translation_op.BackTranslationOp",
            'data_aug_op_config': {
                "model_to": model_class_name,
                "model_back": model_class_name,
                "src_language": "en",
                "tgt_language": "de",
            }
        },
    }

    processor = TmpReplacementDataAugmentProcessor()
    processor.initialize(resources=None, configs=processor_config)

    # pipeline.add(component=processor, config=processor_config)
    pipeline.initialize()

    dataset_path = "data/IMDB_raw/aclImdb/"
    input_file_path = {
        "train": os.path.join(dataset_path, "train"),
        "test": os.path.join(dataset_path, "test")
    }
    output_path = "data/IMDB/"
    output_file_path = {
        "train": os.path.join(output_path, "train.csv"),
        "test": os.path.join(output_path, "test.csv")
    }

    headers = ["content", "label", "id"]

    for split in ["train", "test"]:
        with open(output_file_path[split], "w", encoding="utf-8") as output_file:
            output_file.write("\t".join(headers) + "\n")
            for label in ["pos", "neg"]:
                data_packs: Iterable[MultiPack] = \
                    pipeline.process_dataset(os.path.join(input_file_path[split], label))
                for i, pack in enumerate(data_packs):
                    example_id = pack.get_pack('default').pack_name
                    print(example_id)

                    if split == "train":
                        processor._process(pack)

                    for pack_name in pack.pack_names:
                        p = pack.get_pack(pack_name)
                        for doc in p.get(Document):
                            # currently, the augmented data does not have sentiment label
                            print("Processing " + example_id + ": " + doc.text)
                            output_file.write("\t".join([doc.text, label, example_id]) + "\n")


if __name__ == "__main__":
    main()
