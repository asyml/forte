"""
    Test for generate_ontology.py
"""
import os
import unittest

from forte.data.ontology.generate_ontology import GenerateOntology


class GenerateOntologyTest(unittest.TestCase):
    def setUp(self):
        json_file_path = 'example_ontology.json'
        file_path = GenerateOntology(json_file_path).generate_ontology()
        with open(file_path, 'r') as f:
            self.generated_code = f.read()
        os.remove(file_path)

    def test_generated_code(self):
        with open('new_ontology.py', 'r') as f:
            expected_code =f.read()
        self.assertEqual(self.generated_code, expected_code)
