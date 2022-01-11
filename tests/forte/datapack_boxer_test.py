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
"""
Unit tests for DataPack Boxer.
"""

import os
import unittest
from ddt import ddt, data, unpack
from forte.data.caster import MultiPackBoxer, MultiPackUnboxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline

data_samples_root = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        *([os.path.pardir] * 2),
        "data_samples"
    )
)

onto_specs_samples_root = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        *([os.path.pardir] * 1),
        "forte",
        "data",
        "ontology",
        "test_specs"
    )
)


@ddt
class MultiPackUnboxerTest(unittest.TestCase):
    def test_multi_pack_to_data_pack_unboxer(self):
        from forte.data.readers import OntonotesReader

        # Define and config the Pipeline for MultiPackBoxer test
        nlp_1 = Pipeline[DataPack]()
        nlp_1.set_reader(OntonotesReader())
        pack_name = "test_pack"
        nlp_1.add(MultiPackBoxer(), {"pack_name": pack_name})
        nlp_1.initialize()

        # Define and config the Pipeline for DataPackBoxer test
        nlp_2 = Pipeline[DataPack]()
        nlp_2.set_reader(OntonotesReader())
        pack_name = "test_pack"
        nlp_2.add(MultiPackBoxer(), {"pack_name": pack_name})
        nlp_2.add(MultiPackUnboxer())
        nlp_2.initialize()

        dataset_path = data_samples_root + "/ontonotes/one_file"
        expected_pack_name_multi = "bn/abc/00/abc_0059_multi"
        expected_pack_name = "bn/abc/00/abc_0059"

        # check that the MultiPack is yielded
        pack_1 = nlp_1.process(dataset_path)
        self.assertEqual(pack_1.pack_name, expected_pack_name_multi)
        self.assertTrue(isinstance(pack_1, MultiPack))

        # check that the unboxed DataPack is yielded from the corresponding MultiPack
        pack_2 = nlp_2.process(dataset_path)
        self.assertEqual(pack_2.pack_name, expected_pack_name)
        self.assertTrue(isinstance(pack_2, DataPack))


if __name__ == "__main__":
    unittest.main()
