import json
import unittest
import os
from typing import Iterator

from forte.data.readers import RACEMultiChoiceQAReader
from forte.data.data_pack import DataPack
from ft.onto.race_mutli_choice_qa_ontology_bak import Document, Question


class RACEMultiChoiceQAReaderTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data_samples/race_multi_choice_qa')

    def test_reader_no_replace_test(self):
        # Read with no replacements
        reader = RACEMultiChoiceQAReader()
        data_packs: Iterator[DataPack] = reader.iter(self.dataset_path)
        file_paths: Iterator[str] = reader._collect(self.dataset_path)

        count_packs = 0
        for pack, file_path in zip(data_packs, file_paths):
            count_packs += 1
            expected_text: str = ""
            with open(file_path, "r", encoding="utf8", errors='ignore') as file:
                expected = json.load(file)

            articles = list(pack.get_entries(Document))
            self.assertEqual(len(articles), 1)
            expected_article = expected['article']
            self.assertEqual(articles[0].text, expected_article)
            expected_text += expected_article

            for qid, question in enumerate(pack.get_entries(Question)):
                expected_question = expected['questions'][qid]
                self.assertEqual(question.text, expected_question)
                expected_answers = expected['answers'][qid]
                if not isinstance(expected_answers, list):
                    expected_answers = [expected_answers]
                expected_answers = [reader._convert_to_int(ans)
                                    for ans in expected_answers]
                self.assertEqual(question.answers, expected_answers)
                expected_text += '\n' + expected_question

                for oid, option in enumerate(question.get_options()):
                    expected_option = expected['options'][qid][oid]
                    self.assertEqual(option.text, expected_option)
                    expected_text += '\n' + expected_option

            self.assertEqual(pack.text, expected_text)
        self.assertEqual(count_packs, 2)


if __name__ == "__main__":
    unittest.main()
