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

import logging
import os
from typing import Iterator, Any
from ft.onto.base_ontology import Token, Sentence, Document, EntityMention
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader

__all__ = ["CoNLL03Reader"]


class CoNLL03Reader(PackReader):
    r""":class:`CoNLL03Reader` is designed to read in the CoNLL03 dataset.

    The dataset is from the following paper:
    `Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003
    shared task: Language-independent named entity recognition."
    arXiv preprint cs/0306050 (2003).`

    Data could be downloaded from
    https://deepai.org/dataset/conll-2003-english

    Data format:
    Data files contains one line "-DOCSTART- -X- -X- O" to represent the
    start of a document. After that, each line will contain one word and
    an empty line represent the start of a new sentence. Each line contains
    four fields, the word, its part-of-speech tag, its chunk tag and its
    named entity tag.

    Example:
        EU NNP B-NP B-ORG
        rejects VBZ B-VP O
        German JJ B-NP B-MISC
        call NN I-NP O
        to TO B-VP O
        boycott VB I-VP O
        British JJ B-NP B-MISC
        lamb NN I-NP O
        . . O O
    """

    def _collect(self, conll_directory) -> Iterator[Any]:
        r"""Iterator over conll files in the data_source.

        Args:
            conll_directory: directory to the conll files.

        Returns: Iterator over files in the path with conll extensions.
        """
        logging.info("Reading .conll from %s", conll_directory)
        return dataset_path_iterator(conll_directory, "conll")

    def _cache_key_function(self, collection: str) -> str:
        return os.path.basename(collection)

    def _parse_pack(self, collection: str) -> Iterator[DataPack]:
        with open(collection, "r", encoding="utf8") as doc:
            pack_id: int = 0

            pack: DataPack = DataPack()
            text: str = ""
            offset: int = 0
            has_rows: bool = False

            sentence_begin: int = 0
            sentence_cnt: int = 0

            # NER tag is either "O" or in the format "X-Y",
            # where X is one of B, I,
            # Y is a tag like ORG, PER etc
            prev_y = None
            prev_x = None
            start_index = -1

            for line in doc:
                line = line.strip()

                if line.find("DOCSTART") != -1:
                    # Skip the first DOCSTART.
                    if offset == 0:
                        continue
                    # Add remaining sentence.
                    if has_rows:
                        # Add the last sentence if exists.
                        Sentence(pack, sentence_begin, offset - 1)
                        sentence_cnt += 1

                    pack.set_text(
                        text, replace_func=self.text_replace_operation
                    )
                    Document(pack, 0, len(text))
                    pack.pack_name = f"{collection}_{pack_id}"
                    pack_id += 1
                    yield pack

                    # Create a new datapack.
                    pack = DataPack()
                    text = ""
                    offset = 0
                    has_rows = False

                    sentence_begin = 0
                    sentence_cnt = 0

                    prev_y = None
                    prev_x = None
                    start_index = -1

                elif line != "" and not line.startswith("#"):
                    conll_components = line.split()

                    word = conll_components[0]
                    pos = conll_components[1]
                    chunk_id = conll_components[2]

                    ner_tag = conll_components[3]

                    # A new ner tag occurs.
                    if ner_tag == "O" or ner_tag.split("-")[0] == "B":
                        # Add previous ner tag to sentence if it exists.
                        if prev_y is not None:
                            entity_mention = EntityMention(
                                pack, start_index, offset - 1
                            )
                            entity_mention.ner_type = prev_y

                        # Start process current ner tag.
                        if ner_tag == "O":
                            # Current ner tag is O, reset information.
                            prev_x = None
                            prev_y = None
                            start_index = -1
                        else:
                            # Current ner tag is B.
                            prev_x = "B"
                            prev_y = ner_tag.split("-")[1]
                            start_index = offset
                    # This ner tag is connected to previous one.
                    else:
                        x, y = ner_tag.split("-")
                        assert x == "I", f"Unseen tag {x} in the file."
                        assert y == prev_y, f"Error in {ner_tag}."
                        assert prev_x in ("B", "I"), "Error in {ner_tag}."
                        prev_x = "I"

                    word_begin = offset
                    word_end = offset + len(word)

                    # Add tokens.
                    token = Token(pack, word_begin, word_end)
                    token.pos = pos
                    token.chunk = chunk_id

                    text += word + " "
                    offset = word_end + 1
                    has_rows = True
                else:
                    if not has_rows:
                        # Skip consecutive empty lines.
                        continue
                    # Add sentence
                    Sentence(pack, sentence_begin, offset - 1)

                    # Handle the last ner tag if exists.
                    if prev_x is not None:
                        entity_mention = EntityMention(
                            pack, start_index, offset - 1
                        )
                        entity_mention.ner_type = prev_y

                    # Reset information.
                    sentence_cnt += 1
                    has_rows = False
                    prev_y = None
                    prev_x = None
                    sentence_begin = offset

            if has_rows:
                # Add the last sentence if exists.
                Sentence(pack, sentence_begin, offset - 1)
                sentence_cnt += 1

            pack.set_text(text, replace_func=self.text_replace_operation)
            Document(pack, 0, len(text))
            pack.pack_name = os.path.basename(collection)

            yield pack
