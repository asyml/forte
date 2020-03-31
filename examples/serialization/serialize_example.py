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
from forte.pipeline import Pipeline
from forte.processors.base import MultiPackProcessor
from forte.processors.nltk_processors import (
    NLTKWordTokenizer, NLTKPOSTagger, NLTKSentenceSegmenter)
from forte.processors.writers import DocIdJsonPackWriter, DocIdMultiPackWriter


class PackCopier(MultiPackProcessor):
    """
    Copy the text from existing pack to a new pack.
    """

    def _process(self, input_pack: MultiPack):
        copy_pack: DataPack = DataPack()
        from_pack: DataPack = input_pack.get_pack(self.configs.copy_from)

        copy_pack.set_text(from_pack.text)

        if from_pack.meta.doc_id is not None:
            copy_pack.meta.doc_id = from_pack.meta.doc_id + '_copy'
        else:
            copy_pack.meta.doc_id = 'copy'

        input_pack.add_pack(copy_pack, self.configs.copy_to)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            'copy_from': 'default',
            'copy_to': 'duplicate'
        }


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

    nlp.set_reader(OntonotesReader())
    nlp.add(NLTKSentenceSegmenter())
    nlp.add(NLTKWordTokenizer())
    nlp.add(NLTKPOSTagger())

    # This is a simple writer that serialize the result to the current
    # directory and will use the DocID field in the data pack as the file name.
    nlp.add(
        DocIdJsonPackWriter(),
        {
            'output_dir': output_path,
            'indent': 2,
            'overwrite': True,
        }
    )

    nlp.initialize()
    nlp.run(input_path)
    nlp.finish()


def multi_example(input_path, output_path):
    """
    This example reads data from input path, and write multi pack output
    to output path.

    Args:
        input_path:
        output_path:

    Returns:

    """
    print("Multi Pack serialization example.")

    nlp = Pipeline()
    nlp.set_reader(DirPackReader())

    nlp.add(MultiPackBoxer())

    nlp.add(PackCopier())

    nlp.add(
        DocIdMultiPackWriter(),
        {
            'output_dir': output_path,
            'indent': 2,
            'overwrite': True,
        }
    )

    nlp.initialize()
    nlp.run(input_path)
    nlp.finish()


if __name__ == '__main__':
    data_path = "../../data_samples/ontonotes/00/"
    pack_output = 'pack_out'
    multipack_output = 'multi_out'

    pack_example(data_path, pack_output)
    multi_example(pack_output, multipack_output)
