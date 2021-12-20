# Copyright 2021 The Forte Authors. All Rights Reserved.
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
import tempfile
import unittest
from typing import Dict

from ddt import data, ddt

from forte.data.caster import MultiPackBoxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers import (
    OntonotesReader,
    DirPackReader,
    MultiPackDirectoryReader,
)
from forte.pipeline import Pipeline
from forte.processors.base import (
    MultiPackProcessor,
)
from forte.processors.writers import (
    PackIdJsonPackWriter,
    PackIdMultiPackWriter,
    AutoNamePackWriter,
)
from ft.onto.base_ontology import (
    Sentence,
    EntityMention,
    CrossDocEntityRelation,
)


class CopySentence(MultiPackProcessor):
    """
    Copy the content from existing pack to a new pack.
    """

    def _process(self, input_pack: MultiPack):
        from_pack: DataPack = input_pack.get_pack(self.configs.copy_from)
        copy_pack: DataPack = input_pack.add_pack(self.configs.copy_to)

        copy_pack.set_text(from_pack.text)

        if from_pack.pack_name is not None:
            copy_pack.pack_name = from_pack.pack_name + "_copy"
        else:
            copy_pack.pack_name = "copy"

        s: Sentence
        for s in from_pack.get(Sentence):
            Sentence(copy_pack, s.begin, s.end)

        e: EntityMention
        for e in from_pack.get(EntityMention):
            EntityMention(copy_pack, e.begin, e.end)

    @classmethod
    def default_configs(cls) -> Dict[str, str]:
        return {"copy_from": "default", "copy_to": "duplicate"}


class NaiveCoref(MultiPackProcessor):
    def _process(self, input_pack: MultiPack):
        fp = input_pack.get_pack_at(0)
        sp = input_pack.get_pack_at(1)

        nes1 = list(fp.get(EntityMention))
        nes2 = list(sp.get(EntityMention))

        for ne1 in nes1:
            for ne2 in nes2:
                if ne1.text == ne2.text:
                    CrossDocEntityRelation(input_pack, ne1, ne2)


@ddt
class SerializationTest(unittest.TestCase):
    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        self.data_path = os.path.join(
            file_dir_path, "../../../", "data_samples", "ontonotes", "00"
        )

    @data(
        (True, "pickle"),
        (False, "pickle"),
        (True, "jsonpickle"),
        (False, "jsonpickle"),
    )
    def testMultiPackWriting(self, config_data):
        zip_pack, method = config_data

        # Use different sub-directory to avoid conflicting.
        subdir = f"{zip_pack}_{method}"

        with tempfile.TemporaryDirectory() as main_output:
            # Prepare input data.
            prepared_input: str = os.path.join(
                main_output, subdir, "input_packs"
            )
            data_output: str = os.path.join(main_output, subdir, "output")
            suffix = ".pickle" if method == "pickle" else ".json"
            if zip_pack:
                suffix = suffix + ".gz"

            nlp = Pipeline[DataPack]()
            nlp.set_reader(OntonotesReader())
            nlp.add(
                PackIdJsonPackWriter(),
                {
                    "output_dir": prepared_input,
                    "overwrite": True,
                    "serialize_method": method,
                    "zip_pack": zip_pack,
                },
            )
            nlp.run(self.data_path)

            # Convert to multi pack.
            coref_pl = Pipeline()

            coref_pl.set_reader(
                DirPackReader(),
                {
                    "serialize_method": method,
                    "zip_pack": zip_pack,
                    "suffix": suffix,
                },
            )
            coref_pl.add(MultiPackBoxer())
            coref_pl.add(CopySentence())
            coref_pl.add(NaiveCoref())

            coref_pl.add(
                PackIdMultiPackWriter(),
                config={
                    "output_dir": data_output,
                    "overwrite": True,
                    "serialize_method": method,
                    "zip_pack": zip_pack,
                },
            )
            coref_pl.run(prepared_input)

            self.assertTrue(
                os.path.exists(os.path.join(data_output, "multi.idx"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(data_output, "pack.idx"))
            )
            self.assertTrue(os.path.exists(os.path.join(data_output, "packs")))
            self.assertTrue(os.path.exists(os.path.join(data_output, "multi")))

            # Read the multi pack again.
            mp_pipeline = Pipeline()

            mp_pipeline.set_reader(
                MultiPackDirectoryReader(),
                config={
                    "suffix": suffix,
                    "zip_pack": zip_pack,
                    "serialize_method": method,
                    "data_pack_dir": os.path.join(data_output, "packs"),
                    "multi_pack_dir": os.path.join(data_output, "multi"),
                },
            ).initialize()

            re: CrossDocEntityRelation
            for mp in mp_pipeline.process_dataset():
                for re in mp.get(CrossDocEntityRelation):
                    self.assertEqual(re.get_parent().text, re.get_child().text)

    @data(
        (True, "pickle"),
        (False, "pickle"),
        (True, "jsonpickle"),
        (False, "jsonpickle"),
    )
    def testPackWriting(self, config_data):
        zip_pack, method = config_data

        with tempfile.TemporaryDirectory() as main_output:
            write_pipeline = Pipeline[DataPack]()
            write_pipeline.set_reader(OntonotesReader())
            write_pipeline.add(
                AutoNamePackWriter(),
                {
                    "output_dir": os.path.join(main_output, "packs"),
                    "overwrite": True,
                    "zip_pack": zip_pack,
                    "serialize_method": method,
                },
            )
            write_pipeline.run(self.data_path)

            read_pipeline = Pipeline[DataPack]()
            read_pipeline.set_reader(DirPackReader())
