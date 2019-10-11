"""
    Test for _generate_ontology.py
"""
import os
import pathlib
import unittest

from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator


class GenerateOntologyTest(unittest.TestCase):

    def test_generated_code(self):
        # read json and generate code in a file
        generator = OntologyCodeGenerator()
        json_file_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../configs/example_ontology_config.json')
        json_file_path = str(pathlib.Path(json_file_name).resolve())
        folder_path = generator.generate_ontology(json_file_path,
                                                  is_dry_run=True)

        # record code
        generated_files = []
        final_root = None
        for root, dirs, files in os.walk(folder_path):
            generated_files.extend([os.path.join(root, file)
                                    for file in files if file.endswith('.py')])
            if not root.endswith('__pycache__'):
                final_root = root
        generated_files = sorted(generated_files)

        expected_final_root = os.path.join(folder_path, 'forte/data/ontology')
        self.assertEquals(final_root, expected_final_root)

        file_names = sorted(['example_import_ontology',
                             'example_ontology'])

        expected_files = [f"{os.path.join(final_root, file)}.py"
                          for file in file_names]

        self.assertCountEqual(generated_files, expected_files)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        for i, generated_file in enumerate(generated_files):
            with open(generated_file, 'r') as f:
                generated_code = f.read()

            # assert if generated code matches with the expected code
            expected_code_path = f"{os.path.join(dir_path, file_names[i])}.py"
            with open(expected_code_path, 'r') as f:
                expected_code = f.read()
            self.assertEqual(generated_code, expected_code)
