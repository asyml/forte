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
The reader that reads CoNLL ner_data into our internal json data datasets.
"""
import codecs
import logging
import os
from typing import Iterator, Any

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Token, Sentence, Document

__all__ = [
    "CoNLL03Reader"
]


class CoNLL03Reader(PackReader):
    """
    :class:`CoNLL03Reader` is designed to read in the CoNLL03-ner dataset.
    """

    def _collect(self, conll_directory) -> Iterator[Any]:  # type: ignore
        """
        Iterator over conll files in the data_source

        Args:
            conll_directory: directory to the conll files.

        Returns: Iterator over files in the path with conll extensions.
        """
        logging.info("Reading .conll from %s", conll_directory)
        return dataset_path_iterator(conll_directory, "conll")

    def _cache_key_function(self, conll_file: str) -> str:
        return os.path.basename(conll_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = DataPack()
        doc = codecs.open(file_path, "r", encoding="utf8")

        text = ""
        offset = 0
        has_rows = False

        sentence_begin = 0
        sentence_cnt = 0

        for line in doc:
            line = line.strip()

            if line != "" and not line.startswith("#"):
                conll_components = line.split()

                word = conll_components[1]
                pos = conll_components[2]
                chunk_id = conll_components[3]
                ner_tag = conll_components[4]

                word_begin = offset
                word_end = offset + len(word)

                # Add tokens.
                kwargs_i = {"pos": pos,
                            "chunk": chunk_id,
                            "ner": ner_tag}
                token = Token(pack, word_begin, word_end)

                token.set_fields(**kwargs_i)
                pack.add_or_get_entry(token)

                text += word + " "
                offset = word_end + 1
                has_rows = True
            else:
                if not has_rows:
                    # Skip consecutive empty lines.
                    continue
                # add sentence
                sent = Sentence(pack, sentence_begin, offset - 1)
                pack.add_or_get_entry(sent)

                sentence_begin = offset
                sentence_cnt += 1
                has_rows = False

        if has_rows:
            # Add the last sentence if exists.
            sent = Sentence(pack, sentence_begin, offset - 1)
            sentence_cnt += 1
            pack.add_or_get_entry(sent)

        document = Document(pack, 0, len(text))
        pack.add_or_get_entry(document)

        pack.set_text(text, replace_func=self.text_replace_operation)
        pack.meta.doc_id = file_path
        doc.close()

        yield pack

    @staticmethod
    def default_configs():
        return {}
