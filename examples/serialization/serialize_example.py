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
from forte.processors.writers import DocIdJsonPackWriter
from ft.onto.base_ontology import EntityMention, CrossDocEntityRelation


class PackCopier(MultiPackProcessor):
    """
    Copy the text from existing pack to a new pack.
    """

    def _process(self, input_pack: MultiPack):
        from_pack: DataPack = input_pack.get_pack(self.configs.copy_from)
        copy_pack: DataPack = input_pack.add_pack(self.configs.copy_to)

        copy_pack.set_text(from_pack.text)

        if from_pack.doc_id is not None:
            copy_pack.doc_id = from_pack.doc_id + '_copy'
        else:
            copy_pack.doc_id = 'copy'

        ent: EntityMention
        for ent in from_pack.get(EntityMention):
            EntityMention(copy_pack, ent.begin, ent.end)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            'copy_from': 'default',
            'copy_to': 'duplicate'
        }


class ExampleCoreferencer(MultiPackProcessor):
    """
    Mark some example coreference relations.
    """

    def _process(self, input_pack: MultiPack):
        pack_i = input_pack.get_pack('default')
        pack_j = input_pack.get_pack('duplicate')

        for ent_i, ent_j in zip(pack_i.get(EntityMention),
                                pack_j.get(EntityMention)):
            link = CrossDocEntityRelation(input_pack, ent_i, ent_j)
            link.rel_type = 'coreference'
            input_pack.add_entry(link)


class ExampleCorefCounter(MultiPackProcessor):
    def __init__(self):
        super().__init__()
        self.coref_count = 0

    def _process(self, input_pack: MultiPack):
        rels = input_pack.get_entries_by_type(CrossDocEntityRelation)
        self.coref_count += len(rels)

    def finish(self, _):
        print(f"Found {self.coref_count} pairs in the multi packs.")


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

    nlp.run(input_path)


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

    print("We first read the data, and add multi-packs to them, and then "
          "save the results.")
    coref_pl = Pipeline()
    coref_pl.set_reader(DirPackReader())
    coref_pl.add(MultiPackBoxer())
    coref_pl.add(PackCopier())
    coref_pl.add(ExampleCoreferencer())
    coref_pl.add(ExampleCorefCounter())

    coref_pl.add(
        MultiPackWriter(),
        {
            'output_dir': output_path,
            'indent': 2,
            'overwrite': True,
        }
    )

    coref_pl.run(input_path)

    print("We can then load the saved results, and see if everything is OK. "
          "We should see the same number of multi packs there. ")
    reading_pl = Pipeline()
    reading_pl.set_reader(MultiPackDiskReader(), {'data_path': output_path})
    reading_pl.add(ExampleCorefCounter())
    reading_pl.run()


def main(data_path: str):
    pack_output = 'pack_out'
    multipack_output = 'multi_out'

    pack_example(data_path, pack_output)
    multi_example(pack_output, multipack_output)


if __name__ == '__main__':
    main("../../data_samples/ontonotes/00/")
