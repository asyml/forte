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
A reader to read passages from `MS MARCO` dataset, pertaining to the
Passage Ranking task. Uses the document text for indexing.

Official webpage -
https://github.com/microsoft/MSMARCO-Passage-Ranking#data-information-and-formating
Dataset download link -
https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
Dataset Paper -
Nguyen, Tri, et al. "MS MARCO: A Human-Generated MAchine Reading
COmprehension Dataset." (2016).
"""

import os
from typing import Iterator, Tuple

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Document

__all__ = [
    "MSMarcoPassageReader"
]


class MSMarcoPassageReader(PackReader):
    def __init__(self):
        super().__init__()
        self.configs = None

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable = unused-argument
        self.configs = configs

    def _collect(self, *args, **kwargs) -> Iterator[Tuple[str, str]]:
        # pylint: disable = unused-argument, undefined-variable
        dir_path: str = args[0]

        corpus_file_path = os.path.join(dir_path, 'collection.tsv')

        with open(corpus_file_path, 'r') as file:
            for line in file:
                doc_id, doc_content = line.split('\t', 1)
                yield doc_id, doc_content

    def _parse_pack(self, doc_info: Tuple[str, str]) -> Iterator[DataPack]:
        r"""Takes the `doc_info` returned by the `_collect` method and returns a
        `data_pack` that either contains entry of the type `Query`, or contains
        an entry of the type Document.

        Args:
            doc_info: document info to be populated in the data_pack.

        Returns: query or document data_pack.
        """
        data_pack: DataPack = DataPack()

        doc_id, doc_text = doc_info
        data_pack.meta.doc_id = doc_id

        # add documents
        data_pack.add_or_get_entry(Document(data_pack, 0, len(doc_text)))
        data_pack.set_text(doc_text)

        yield data_pack

    def _cache_key_function(self, data_pack: DataPack) -> str:
        if data_pack.meta.doc_id is None:
            raise ValueError("Data pack does not have a document id.")
        return data_pack.meta.doc_id
