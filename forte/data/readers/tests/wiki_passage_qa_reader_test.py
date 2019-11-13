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
Tests for wiki_passage_qa_reader.
"""
import os
import json
import unittest
from typing import List, Dict
from ddt import ddt, data

from texar.torch import HParams

from ft.onto.base_ontology import Query, Document

from forte.common import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.wiki_passage_qa_reader import WikiPassageQAReader


@ddt
class WikiPassageQAReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = WikiPassageQAReader()

        self.data_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data_samples/wiki_passage_qa')

        self.expected_queries = [
            "Why have Canadians more successfully adopted multiculturalism?",
            "How did French become prominent?",
            "How did interlinked metropolitan areas come to be described?",
            "What were the differences between men and women's roles in "
            "Paleolithic society?",
            "What were the effects of the early 20th century on New Zealand?"
        ]

        self.expected_doc_ids = {
            "passage": [["89_8"],
                        ["684_15", "684_16"],
                        ["492_12"],
                        ["462_27"],
                        ["45_11", "45_12"]],
            "document": [["89"],
                         ["684"],
                         ["492"],
                         ["462"],
                         ["45"]]}

        corpus_file = os.path.join(self.data_dir, 'document_passages.json')
        with open(corpus_file, 'r') as f:
            self.expected_docs_text = json.loads(f.read())

    @data("passage", "document")
    def test_document_and_passage_mode(self, doc_mode):
        resources: Resources = Resources()
        config: HParams = HParams({"doc_mode": doc_mode},
                                  default_hparams=None)
        self.reader.initialize(resources, config)
        data_packs: List[DataPack] = \
            [data_pack for data_pack in self.reader.iter(self.data_dir,
                                                         'dev')]

        # get all queries and all documents
        queries: List[Query] = []
        documents: Dict[str, Document] = dict()
        for data_pack in data_packs:
            query_entries = list(data_pack.get_entries_by_type(Query))
            doc_entries = list(data_pack.get_entries_by_type(Document))

            self.assertTrue(len(query_entries) + len(doc_entries) == 1)

            if len(query_entries) > 0:
                query_entry: Query = query_entries[0]
                queries.append(query_entry)
            else:
                doc_entry: Document = doc_entries[0]
                documents[data_pack.meta.doc_id] = doc_entry

        # match text of documents relevant to the queries to the actual text
        for i, query in enumerate(queries):
            expected_query = self.expected_queries[i]
            expected_ids = self.expected_doc_ids[doc_mode][i]
            self.assertEqual(query.query, expected_query)
            self.assertCountEqual(query.doc_ids["relevant_docs"],
                                  expected_ids)
            for doc_id in expected_ids:
                expected_text = self.get_expected_text(doc_id, doc_mode)
                self.assertEqual(documents[doc_id].text, expected_text)

    def get_expected_text(self, doc_id, doc_mode):
        if doc_mode == "passage":
            did, pid = doc_id.split('_')
            return self.expected_docs_text[did][pid]
        document = self.expected_docs_text[doc_id]
        passages = [document[pid]for pid in sorted(document.keys())]
        return os.linesep.join(passages)
