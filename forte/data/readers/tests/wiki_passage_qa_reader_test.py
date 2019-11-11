"""
Tests for wiki_passage_qa_reader.
"""
import os
import json
import unittest
from typing import List, Dict
from texar.torch import HParams

from ft.onto.base_ontology import Query, Document

from forte.common import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.wiki_passage_qa_reader import WikiPassageQAReader


class WikiPassageQAReaderTest(unittest.TestCase):
    def setUp(self):
        reader = WikiPassageQAReader()
        resources: Resources = Resources()
        config: HParams = HParams({"doc_mode": "passage"}, default_hparams=None)
        reader.initialize(resources, config)

        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'data_samples/wiki_passage_qa')

        self.data_packs: List[DataPack] = \
            [data_pack for data_pack in reader.iter(data_dir, 'dev')]

        self.expected_queries = [
            ("Why have Canadians more successfully adopted multiculturalism?",
             ["89_8"]),
            ("How did French become prominent?", ["684_15", "684_16"]),
            ("How did interlinked metropolitan areas come to be described?",
             ["492_12"]),
            ("What were the differences between men and women's roles in "
             "Paleolithic society?", ["462_27"]),
            ("What were the effects of the early 20th century on New Zealand?",
             ["45_11", "45_12"])
        ]
        with open(os.path.join(data_dir, 'document_passages.json'), 'r') as f:
            self.expected_docs_text = json.loads(f.read())

    def test_reader_text(self):
        # get all queries and all documents
        queries: List[Query] = []
        documents: Dict[str, Document] = dict()
        for data_pack in self.data_packs:
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
            expected_query, expected_ids = self.expected_queries[i]
            self.assertEqual(query.query, expected_query)
            self.assertCountEqual(query.doc_ids["relevant_docs"], expected_ids)
            for doc_id in expected_ids:
                did, pid = doc_id.split('_')
                self.assertEqual(documents[doc_id].text,
                                 self.expected_docs_text[did][pid])
