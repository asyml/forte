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
from typing import Iterator, Iterable, List

from forte.data.readers import OpenIEReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, PredicateMention, Document, \
    PredicateArgument, PredicateLink, Token


class OpenIEReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the pipeline.
        self.dataset_path: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/openie'))

        self.pipeline: Pipeline = Pipeline[DataPack]()
        self.reader: OpenIEReader = OpenIEReader()
        self.pipeline.set_reader(self.reader)
        self.pipeline.initialize()

    def test_process_next(self):
        data_packs: Iterable[DataPack] = self.pipeline.process_dataset(
            self.dataset_path)
        file_paths: Iterator[str] = self.reader._collect(self.dataset_path)

        count_packs: int = 0

        for pack, file_path in zip(data_packs, file_paths):
            count_packs += 1
            expected_doc: str = ""
            with open(file_path, "r", encoding="utf8", errors='ignore') as file:
                expected_doc = file.read()

            # Test document.
            actual_docs: List[Document] = list(pack.get(Document))
            self.assertEqual(len(actual_docs), 1)
            actual_doc: Document = actual_docs[0]
            self.assertEqual(actual_doc.text,
                             expected_doc.replace('\t', ' ').replace('\n', ' ')
                             + ' ')

            lines: List[str] = expected_doc.split('\n')
            actual_sentences: Iterator[Sentence] = pack.get(Sentence)
            actual_predicates: Iterator[PredicateMention] = \
                pack.get(PredicateMention)
            actual_args: Iterator[PredicateArgument] = \
                pack.get(PredicateArgument)
            # Force sorting as Link entries have no order when retrieving from
            # data pack.
            actual_link_ids: Iterator[int] = \
                iter(sorted(pack.get_ids_by_type(PredicateLink)))

            for line in lines:
                line: str = line.strip()
                line: List[str] = line.split('\t')

                # Test sentence.
                expected_sentence: str = line[0]
                actual_sentence: Sentence = next(actual_sentences)
                self.assertEqual(actual_sentence.text, expected_sentence)

                actual_full_predicate: PredicateMention = \
                    next(actual_predicates)
                actual_head_predicate: Token = actual_full_predicate.headword

                # Test head predicate.
                expected_head_predicate: str = line[1]
                self.assertEqual(actual_head_predicate.text,
                                 expected_head_predicate)

                # Test full predicate.
                expected_full_predicate: str = line[2]
                self.assertEqual(actual_full_predicate.text,
                                 expected_full_predicate)

                # Test argument.
                for expected_arg in line[3:]:
                    actual_arg: PredicateArgument = next(actual_args)
                    self.assertEqual(actual_arg.text, expected_arg)

                    # Test predicate relation link.
                    actual_link: PredicateLink = \
                        pack.get_entry(next(actual_link_ids))
                    self.assertEqual(actual_link.get_parent().text,
                                     expected_full_predicate)
                    self.assertEqual(actual_link.get_child().text, expected_arg)

        self.assertEqual(count_packs, 1)


if __name__ == '__main__':
    unittest.main()
