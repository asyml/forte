"""
A reader to read `WikiPassageQA` dataset, that contains documents divided into
passages and queries labeled with relevant passages and documents. This reader
yields query and documents as individual data_packs.

Dataset paper: Cohen, Daniel, Liu Yang, and W. Bruce Croft. "WikiPassageQA: A
benchmark collection for research on non-factoid answer passage retrieval." In
The 41st International ACM SIGIR Conference on Research & Development in
Information Retrieval.
Dataset download link: https://ciir.cs.umass.edu/downloads/wikipassageqa/
"""
import os
import json

from typing import Iterator, List, Tuple, Optional, Union

import pandas as pd
from texar.torch import HParams

from ft.onto.base_ontology import Query, Document, Passage
from forte.data import DataPack
from forte.data.readers.base_reader import PackReader
from forte.common.resources import Resources

__all__ = [
    "WikiPassageQAReader"
]


class WikiPassageQAReader(PackReader):

    DocInfoType = Tuple[bool, str, List[str], Optional[List[str]]]

    def __init__(self):
        super(WikiPassageQAReader, self).__init__()
        self.configs = None

    def initialize(self, resources: Resources, configs: HParams):
        # pylint: disable = unused-argument
        self.configs = configs

    def _collect(self, *args, **kwargs) -> Iterator[DocInfoType]:
        # pylint: disable = unused-argument, undefined-variable
        """
        Reads the contents of the input `dir_path` and returns a info to
        populate query or document data packs. It reads the documents from the
        json files, and queries with their relevant documents from the
        query-relevant_document pairs from the file corresponding to the `split`

        As `WikipediaPassageQa` provides both document and passages that are
        relevant to a query, it is possible to define `doc_mode` through
        `self.config` that could either be "passage" or "document" based on the
        granularity of the retrieval desired.

        :param dir_path: Directory path of the
        :param split: 'train', 'dev' or 'test' to refer to the query-document
        labels of the given split type.
        :return: yields - (1) type of the data_pack (query or document)
                          (2) unique id of the query or document
                          (3) text lines of the query or document
                          (multiple lines for documents when
                          self.configs.doc_mode == "passage", otherwise contains
                          just one line of text)
                          (4) documents associated with a query in case the
                          data_pack is of type Query
        """
        dir_path: str = args[0]
        split: str = args[1]

        corpus_file_path = os.path.join(dir_path, 'document_passages.json')
        split_file_path = os.path.join(dir_path, f'{split}.tsv')

        passage_mode = self.configs.doc_mode == "passage"

        with open(corpus_file_path, 'r') as f:
            corpus = json.loads(f.read())

        df = pd.read_csv(split_file_path, sep='\t')

        def _get_passage_ids(row_):
            did = f'{row_["DocumentID"]}'
            return [f'{did.strip()}_{pid.strip()}'
                    for pid in f'{row_["RelevantPassages"]}'.split(',')]

        if passage_mode:
            df["labels"] = df.apply(_get_passage_ids, axis=1)
        else:
            df["labels"].apply(lambda labels: labels.split(','), inplace=True)

        # send all the queries with relevant docs
        for _, row in df.iterrows():
            yield True, row["QID"], [row["Question"]], row["labels"]

        # send all the docs
        for _, row in df.iterrows():
            for doc_id in row["labels"]:
                if passage_mode:
                    doc_id_, passage_id = doc_id.split('_')
                    doc_content = [corpus[doc_id_][passage_id]]
                else:
                    doc_content = [corpus[doc_id][passage_id]
                                   for passage_id in corpus[doc_id]]
                yield False, doc_id, doc_content, None

    def _parse_pack(self, doc_info: DocInfoType) -> Iterator[DataPack]:
        # pylint: disable = no-self-use
        """
        Takes the `doc_info` returned by the `_collect` method and returns a
        `data_pack` that either contains entry of the type `Query`, or contains
        an entry of the type Document.
        :param doc_info: document info to be populatd in the data_pack
        :return: query or document data_pack
        """
        data_pack: DataPack = DataPack()

        is_query, doc_id, doc_content, rel_docs = doc_info
        data_pack.meta.doc_id = doc_id

        annotations: List[Union[Passage, Document, Query]] = []

        if not is_query:
            # add passages
            for passage in doc_content:
                annotations.append(Passage(data_pack, 0, len(passage)))

            doc_text = '\n'.join(doc_content)
            annotations.append(Document(data_pack, 0, len(doc_text)))
            data_pack.set_text(doc_text)
        else:
            query = Query(data_pack)
            query.query = doc_content[0]
            if rel_docs is not None:
                query.doc_ids = {"relevant_docs": rel_docs}
            annotations = [query]

        # add annotations to data_pack
        for annotation in annotations:
            data_pack.add_or_get_entry(annotation)

        yield data_pack

    def _cache_key_function(self, data_pack: DataPack) -> str:
        # pylint: disable = no-self-use
        if data_pack.meta.doc_id is None:
            raise ValueError("Data pack does not have a document id")
        return data_pack.meta.doc_id
