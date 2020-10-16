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
Unit tests for OpenIEReader.
"""
import os
import unittest
import json
from typing import Iterator, Iterable, List
from forte.data.readers.qasrl_reader import QASRLReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Document



class QASRLReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the pipeline.
        self.dataset_path: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/qa_srl_bank'))

        self.pipeline: Pipeline = Pipeline[DataPack]()
        self.reader: QASRLReader = QASRLReader()
        self.pipeline.set_reader(self.reader)
        self.pipeline.initialize()

    def test_process_next(self):
        data_packs: Iterable[DataPack] = self.pipeline.process_dataset(self.dataset_path)
        file_paths: Iterator[str] = self.reader._collect(self.dataset_path)

        count_packs: int = 0

        # Each .qa file is corresponding to an Iterable Obj
        for pack, file_path in zip(data_packs, file_paths):

            count_packs += 1
            expected_doc: str = ""

            # Read all lines in .qa file
            with open(file_path, "r", encoding="utf8", errors='ignore') as file:
                expected_doc = file.read()

            # Test document.
            actual_docs: List[Document] = list(pack.get(Document))
            self.assertEqual(len(actual_docs), 1)
            actual_doc: Document = actual_docs[0]

            self.assertEqual(actual_doc.text, expected_doc.replace('\n', ' ') + ' ')

            lines: List[str] = expected_doc.split('\n')
            sen_lines = []
            for line in lines:
                line_json = json.loads(line)
                sentenceTokens = str(line_json["sentenceTokens"])
                sentenceTokens = sentenceTokens.replace(" ", "")
                sentenceTokens = sentenceTokens.replace("'", "\"")
                sen_lines.append(sentenceTokens)

            actual_sentences: Iterator[Sentence] = pack.get(Sentence)
            # Force sorting as Link entries have no order when retrieving from
            # data pack.

            for line, actual_sentence in zip(sen_lines, actual_sentences):
                line: str = line.strip()

                # Test sentence.
                expected_sentence: str = line
                self.assertEqual(actual_sentence.text, expected_sentence)

        self.assertEqual(count_packs, 1)


if __name__ == '__main__':
    unittest.main()
