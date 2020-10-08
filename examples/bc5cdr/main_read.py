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
from typing import Any, Dict

from forte.data.caster import MultiPackBoxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers import OntonotesReader, DirPackReader
from forte.data.readers.deserialize_reader import MultiPackDiskReader
from forte.pipeline import Pipeline
from forte.processors.base import MultiPackProcessor, MultiPackWriter
from forte.processors.nltk_processors import (
    NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter)
from forte.processors.writers import PackNameJsonPackWriter
from ft.onto.base_ontology import EntityMention, CrossDocEntityRelation
from forte.data.readers import bc5cdr_reader


def pack_example(input_path, output_path):
    """
    This example read data from input path and serialize to output path.
    Args:
        input_path:
        output_path:

    Returns:

    """
    print("Pack serialization example.")
    nlp = Pipeline[DataPack]()

    nlp.set_reader((bc5cdr_reader.BC5CDRReader()))

    # This is a simple writer that serialize the result to the current
    # directory and will use the DocID field in the data pack as the file name.
    nlp.add(
        PackNameJsonPackWriter(),
        {
            'output_dir': output_path,
            'indent': 2,
            'overwrite': True,
        }
    )

    nlp.run(input_path)



def main(data_path: str):
    pack_output = 'pack_out'
    # multipack_output = 'multi_out'

    pack_example(data_path, pack_output)

if __name__ == '__main__':
    main("data/train/")
