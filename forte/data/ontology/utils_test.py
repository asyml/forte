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
import os
import unittest
from ddt import ddt, data

import jsonschema

from forte.data.ontology import utils


@ddt
class JsonValidationTest(unittest.TestCase):
    def setUp(self):
        self.valid_filepath = 'data_samples/ontology/validation_schema.json'

    @data(
        "data_samples/ontology/configs/example_ontology_config.json",
        "data_samples/ontology/configs/example_import_ontology_config.json",
        "data_samples/ontology/configs/"
        "example_multi_module_ontology_config.json",
        "data_samples/ontology/configs/example_complex_ontology_config.json",
        "data_samples/ontology/configs/stanfordnlp_ontology.json",
        "data_samples/ontology/test/test_configs/test_composite_item_type.json"
    )
    def test_valid_json(self, input_filepath):
        utils.validate_json_schema(input_filepath, self.valid_filepath)

    @data(
        ("data_samples/ontology/test/test_configs/"
         "test_duplicate_attribute.json",
         "non-unique elements"),
        ("data_samples/ontology/test/test_configs/"
         "test_additional_properties.json",
         "Additional properties are not allowed")
    )
    def test_invalid_json(self, value):
        input_filepath, error_msg = value
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            utils.validate_json_schema(input_filepath, self.valid_filepath)
        self.assertTrue(error_msg in cm.exception.args[0])


if __name__ == '__main__':
    unittest.main()
