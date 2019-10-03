"""
    Test for generate_ontology.py
"""
import os
import unittest

from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator


class GenerateOntologyTest(unittest.TestCase):

    def test_generated_code(self):
        # read json and generate code in a file
        generator = OntologyCodeGenerator()
        curr_path = os.path.dirname(os.path.realpath(__file__))
        json_file_path = os.path.join(curr_path, 'example_ontology_config.json')
        ontology_full_name, file_path, folder_path = \
            generator.generate_ontology(json_file_path)

        # record code
        with open(file_path, 'r') as f:
            self.generated_code = f.read()

        # clean up the generated file and folders
        generator.cleanup_generated_ontology(ontology_full_name)

        # assert if generated code matches with the expected code
        expected_file_path = os.path.join(curr_path, 'true_example_ontology.py')
        with open(expected_file_path, 'r') as f:
            expected_code = f.read()
        self.assertEqual(self.generated_code, expected_code)
