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
import re

from forte.data.caster import MultiPackBoxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers import OntonotesReader, DirPackReader
from forte.pipeline import Pipeline
from forte.processors.base import MultiPackProcessor, MultiPackWriter, \
    PackProcessor
from forte.processors.writers import PackNameJsonPackWriter, \
    PackIdJsonPackWriter, PackNameMultiPackWriter, PackIdMultiPackWriter
from ft.onto.base_ontology import Sentence


class CopySentence(MultiPackProcessor):
    """
    Copy the content from existing pack to a new pack.
    """

    def _process(self, input_pack: MultiPack):
        from_pack: DataPack = input_pack.get_pack(self.configs.copy_from)
        copy_pack: DataPack = input_pack.add_pack(self.configs.copy_to)

        copy_pack.set_text(from_pack.text)

        if from_pack.pack_name is not None:
            copy_pack.pack_name = from_pack.pack_name + '_copy'
        else:
            copy_pack.pack_name = 'copy'

        s: Sentence
        for s in from_pack.get(Sentence):
            Sentence(copy_pack, s.begin, s.end)

    @classmethod
    def default_configs(cls) -> Dict[str, str]:
        return {
            'copy_from': 'default',
            'copy_to': 'duplicate'
        }


class SerializationTest(unittest.TestCase):
    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        data_path = os.path.join(
            file_dir_path, "../../../../", 'data_samples', 'ontonotes', '00')

        self.main_output = tempfile.TemporaryDirectory()

        nlp = Pipeline[DataPack]()
        nlp.set_reader(OntonotesReader())
        nlp.add(
            PackIdJsonPackWriter(),
            {
                'output_dir': os.path.join(self.main_output.name, 'packs'),
                'indent': 2,
                'overwrite': True,
            }
        )
        nlp.run(data_path)

    def testMultiPackWriting(self):
        coref_pl = Pipeline()
        coref_pl.set_reader(DirPackReader())
        coref_pl.add(MultiPackBoxer())
        coref_pl.add(CopySentence())

        coref_pl.add(
            PackIdMultiPackWriter(),
            config={
                'output_dir': os.path.join(self.main_output.name, 'multi'),
                'indent': 2,
                'overwrite': True,
            }
        )
        coref_pl.run(os.path.join(self.main_output.name, 'packs'))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'multi.idx')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'pack.idx')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'packs')))
        self.assertTrue(os.path.exists(os.path.join('multi_out', 'multi')))

    def testStaveFormat(self):
        # TODO: call stave transform here.
        pass

    def tearDown(self):
        self.main_output.cleanup()
