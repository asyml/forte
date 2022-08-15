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
    Tests for the module forte.data.ontology.ontology_code_generator
"""
import importlib
import os
import sys
import tempfile
import unittest
import warnings
from string import Template

import jsonschema
import pytest
from ddt import ddt, data
from testfixtures import LogCapture, log_capture

from jsonschema.exceptions import ValidationError
from forte.data.ontology import utils
from forte.data.ontology.code_generation_exceptions import (
    DuplicatedAttributesWarning,
    DuplicateEntriesWarning,
    OntologySourceNotFoundException,
    TypeNotDeclaredException,
    UnsupportedTypeException,
    ParentEntryNotSupportedException,
    InvalidIdentifierException,
    CodeGenerationException,
)
from forte.data.ontology.code_generation_objects import ImportManager
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator


@ddt
class GenerateOntologyTest(unittest.TestCase):
    def setUp(self):
        self.generator = OntologyCodeGenerator()
        self.dir_path = None

        curr_dir = os.path.dirname(__file__)

        self.spec_dir = os.path.join(curr_dir, "test_specs/")
        self.test_output = os.path.join(curr_dir, "test_outputs/")

    def tearDown(self):
        """
        Cleans up the generated files after test case if any. Only cleans up if
        generate_ontology passes successfully.
        """
        if self.dir_path is not None:
            self.generator.cleanup_generated_ontology(
                self.dir_path, is_forced=True
            )

    def assert_generation_equal(self, file_a, file_b):
        with open(file_a, "r") as a, open(file_b, "r") as b:
            lines_a = a.readlines()
            lines_b = b.readlines()
            self.assertEqual(len(lines_a), len(lines_b))
            for la, lb in zip(lines_a, lines_b):
                # Skip source path line.
                if la.startswith("# ***source json:") and lb.startswith(
                    "# ***source json:"
                ):
                    continue
                self.assertEqual(la, lb)

    @data(
        (
            "example_ontology",
            ["ft/onto/example_import_ontology", "ft/onto/example_ontology"],
        ),
        ("example_complex_ontology", ["ft/onto/example_complex_ontology"]),
        (
            "example_multi_module_ontology",
            ["ft/onto/ft_module", "custom/user/custom_module"],
        ),
        ("race_qa_onto", ["ft/onto/race_qa_ontology"]),
        ("test_top_attribute", ["ft/onto/test_top_attribute"]),
        ("test_ndarray_attribute", ["ft/onto/test_ndarray"])
    )
    def test_generated_code(self, value):
        input_file_name, file_paths = value
        file_paths = sorted(file_paths + _get_init_paths(file_paths))

        # Read json and generate code in a file.
        with tempfile.TemporaryDirectory() as tempdir:
            json_file_path = os.path.join(
                self.spec_dir, f"{input_file_name}.json"
            )
            folder_path = self.generator.generate(
                json_file_path, tempdir, is_dry_run=True
            )
            self.dir_path = folder_path

            # Reorder code.
            generated_files = sorted(
                utils.get_generated_files_in_dir(folder_path)
            )

            expected_files = [
                f"{os.path.join(folder_path, file)}.py" for file in file_paths
            ]

            self.assertEqual(generated_files, expected_files)

            for i, generated_file in enumerate(generated_files):
                # assert if generated code matches with the expected code
                expected_code_path = os.path.join(
                    self.test_output, f"{file_paths[i]}.py"
                )
                self.assert_generation_equal(generated_file, expected_code_path)

    def test_dry_run_false(self):
        json_file_path = os.path.join(
            self.spec_dir, "example_import_ontology.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(json_file_path, temp_dir)
            self.generator.generate(temp_filename, temp_dir, is_dry_run=False)
            folder_path = temp_dir
            for name in ["ft", "onto", "example_import_ontology.py"]:
                self.assertTrue(name in os.listdir(folder_path))
                folder_path = os.path.join(folder_path, name)

    @data(
        (0),
        (1),
        (2),
    )
    def test_namespace_depth(self, namespace_depth):
        json_file_path = os.path.join(
            self.spec_dir, "example_import_ontology.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(json_file_path, temp_dir)
            # Test with namespace_depth = int
            folder_path = self.generator.generate(
                temp_filename,
                temp_dir,
                is_dry_run=False,
                namespace_depth=namespace_depth,
            )
            gen_files = sorted(utils.get_generated_files_in_dir(folder_path))

            exp_file_path_all = [
                "ft/__init__.py",
                "ft/onto/__init__.py",
                "ft/onto/example_import_ontology.py",
            ]
            exp_file_path = exp_file_path_all[namespace_depth:]
            exp_files = sorted(
                [
                    f"{os.path.join(folder_path, file)}"
                    for file in exp_file_path
                ]
            )

            self.assertEqual(gen_files, exp_files)

    @data(
        (True, "test_duplicate_entry.json", DuplicateEntriesWarning, True),
        (True, "test_duplicate_attr_name.json", DuplicatedAttributesWarning, True),
        (True, "test_ndarray_dtype_only.json", UserWarning, True),
        (True, "test_ndarray_shape_only.json", UserWarning, True),
        (True, "test_self_reference.json", UserWarning, False),
        (False, "example_ontology.json", OntologySourceNotFoundException, True),
        (False, "test_invalid_parent.json", ParentEntryNotSupportedException, True),
        (False, "test_invalid_attribute.json", TypeNotDeclaredException, True),
        (False, "test_nested_item_type.json", UnsupportedTypeException, True),
        (False, "test_no_item_type.json", TypeNotDeclaredException, True),
        (False, "test_unknown_item_type.json", TypeNotDeclaredException, True),
        (False, "test_invalid_entry_name.json", InvalidIdentifierException, True),
        (False, "test_invalid_attr_name.json", InvalidIdentifierException, True),
        (False, "test_non_string_keys.json", CodeGenerationException, True),
    )
    def test_warnings_errors(self, value):
        is_warning, file, msg_type, expect_happen = value
        temp_dir = tempfile.mkdtemp()
        json_file_name = os.path.join(self.spec_dir, file)
        temp_filename = _get_temp_filename(json_file_name, temp_dir)
        if is_warning:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.generator.generate(
                    temp_filename, temp_dir, is_dry_run=True
                )
                if expect_happen:
                    self.assertEqual(len(w), 1)
                    assert w[0].category, msg_type
                else:
                    self.assertEqual(len(w), 0)
        else:
            if expect_happen:
                with self.assertRaises(msg_type):
                    self.generator.generate(
                        temp_filename, temp_dir, is_dry_run=True
                    )
            else:
                try:
                    self.generator.generate(
                        temp_filename, temp_dir, is_dry_run=True
                    )
                except msg_type:
                    pytest.fail("Shouldn't raise this exception.")

    @log_capture()
    def test_directory_already_present(self):
        json_file_path = os.path.join(
            self.spec_dir, "example_import_ontology.json"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            os.mkdir(os.path.join(temp_dir, "ft"))
            temp_filename = _get_temp_filename(json_file_path, temp_dir)
            with LogCapture() as lc:
                self.generator.generate(temp_filename, temp_dir, False)
                lc.check_present(
                    (
                        "root",
                        "WARNING",
                        f"The directory with the name ft is already present "
                        f"in {temp_dir}. New files will be merge into the "
                        f"existing directory. Note that in this "
                        f"case, the namespace depth may not take effect.",
                    )
                )

    def test_top_ontology_parsing_imports(self):
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, "temp.py")
        sys.path.append(temp_dir)
        with open(temp_filename, "w") as temp_file:
            temp_file.write(
                "import os.path\n"
                "import os.path as os_path\n"
                "from os import path\n"
            )
        temp_module = importlib.import_module("temp")

        manager = ImportManager(None, None)

        gen = OntologyCodeGenerator()
        gen.initialize_top_entries(manager, temp_module)

        imports = manager.get_import_statements()

        expected_imports = ["from os import path"]

        self.assertListEqual(imports, expected_imports)

    @data(
        "example_ontology.json",
        "example_import_ontology.json",
        "example_multi_module_ontology.json",
        "example_complex_ontology.json",
        "test_unknown_item_type.json",
    )
    def test_valid_json(self, input_filepath):
        input_filepath = os.path.join(self.spec_dir, input_filepath)
        utils.validate_json_schema(input_filepath)

    @data(
        ("test_duplicate_attribute.json", "non-unique elements"),
        (
            "test_additional_properties.json",
            "Additional properties are not allowed",
        ),
    )
    def test_invalid_json(self, value):
        input_filepath, error_msg = value
        input_filepath = os.path.join(self.spec_dir, input_filepath)
        with self.assertRaises(jsonschema.exceptions.ValidationError) as cm:
            utils.validate_json_schema(input_filepath)
        self.assertTrue(error_msg in cm.exception.args[0])

    @data(
        [1],
        [3, ],
        [2, 2],
        [[1, 2], [3, 4]]
    )
    def test_ndarray_valid_shape(self, shape):
        mapping = {
            "dtype": '"int"',
            "shape": f"{shape}"
        }
        template_file = os.path.join(self.spec_dir, "test_ndarray_template.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(template_file, temp_dir)
            _modify_test_template(
                template_file=temp_filename,
                mapping=mapping,
                output_path=temp_filename)
            utils.validate_json_schema(temp_filename)

    @data(
        (False, 3),
        (True, [2, 2])
    )
    def test_ndarray_invalid_shape(self, value):
        is_string, shape = value
        mapping = {
            "dtype": '"int"',
            "shape": '"' + f"{shape}" + '"' if is_string else f"{shape}"
        }
        template_file = "./tests/forte/data/ontology/test_specs/test_ndarray_template.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(template_file, temp_dir)
            _modify_test_template(
                template_file=temp_filename,
                mapping=mapping,
                output_path=temp_filename)
            with self.assertRaises(ValidationError):
                utils.validate_json_schema(temp_filename)

    @data(
        "bool",
        "bool8",
        "int",
        "int8",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "float",
        "float32",
        "float64",
        "float96",
        "float128",
        "complex",
        "complex128",
        "complex192",
        "complex256"
    )
    def test_ndarray_valid_dtype(self, dtype):
        mapping = {
            "dtype": '"' + f"{dtype}" + '"',
            "shape": [2, 2]
        }
        template_file = "./tests/forte/data/ontology/test_specs/test_ndarray_template.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(template_file, temp_dir)
            _modify_test_template(
                template_file=temp_filename,
                mapping=mapping,
                output_path=temp_filename)
            utils.validate_json_schema(temp_filename)

    @data(
        "xint",
        "undefined_dtype"
    )
    def test_ndarray_invalid_dtype(self, dtype):
        mapping = {
            "dtype": '"' + f"{dtype}" + '"',
            "shape": [2, 2]
        }
        template_file = "./tests/forte/data/ontology/test_specs/test_ndarray_template.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = _get_temp_filename(template_file, temp_dir)
            _modify_test_template(
                template_file=temp_filename,
                mapping=mapping,
                output_path=temp_filename)
            with self.assertRaises(ValidationError):
                utils.validate_json_schema(temp_filename)


def _get_temp_filename(json_file_path, temp_dir):
    with open(json_file_path, "r") as f:
        json_content = f.read()
    temp_filename = os.path.join(temp_dir, "temp.json")
    with open(temp_filename, "w") as temp_file:
        temp_file.write(json_content)
    return temp_filename


def _get_init_paths(paths):
    inits = set()
    for path in paths:
        tmp_path = path
        for _ in range(len(path.split("/")) - 1):
            tmp_path = tmp_path.rsplit("/", 1)[0]
            inits.add(os.path.join(tmp_path, "__init__"))
    return list(inits)


def _modify_test_template(template_file, mapping, output_path):
    """
    This helper function takes in a template of ontology config
    and a mapping to substitute key words in the template.

    Args:
        template_file (str): path to the template JSON file.
        mapping (dict): mapping to substitute key words.
        output_path (str): output path of the generated file.
    """
    with open(template_file, 'r') \
            as template_file:
        data = template_file.read()
    data = Template(data)
    data = data.substitute(mapping)
    with open(output_path, 'w') \
            as output_json:
        output_json.write(data)
