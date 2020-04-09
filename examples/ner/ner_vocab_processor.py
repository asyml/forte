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
from collections import Counter

import numpy as np
import torch

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.models.ner.utils import load_glove_embedding, normalize_digit_word
from forte.processors import Alphabet
from forte.processors import VocabularyProcessor
from ft.onto.base_ontology import Token, Sentence

__all__ = [
    "CoNLL03VocabularyProcessor",
]

logging.basicConfig(level=logging.DEBUG)


def construct_word_embedding_table(embed_dict, alphabet):
    embedding_dim = list(embed_dict.values())[0].shape[-1]

    scale = np.sqrt(3.0 / embedding_dim)
    table = np.empty(
        [alphabet.size(), embedding_dim], dtype=np.float32
    )
    oov = 0
    for word, index in alphabet.items():
        if word in embed_dict:
            embedding = embed_dict[word]
        elif word.lower() in embed_dict:
            embedding = embed_dict[word.lower()]
        else:
            embedding = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            ).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    return torch.from_numpy(table)


class CoNLL03VocabularyProcessor(VocabularyProcessor):
    """
    Vocabulary Processor for the datasets of CoNLL data
    Create the vocabulary for the word, character, pos tag, chunk id and ner
    tag
    """

    def __init__(self) -> None:
        super().__init__()
        self.normalize_digit: bool = True
        self.embedding_path: str = '.'

        self.word_cnt: Counter = Counter()
        self.char_cnt: Counter = Counter()
        self.pos_cnt: Counter = Counter()
        self.chunk_cnt: Counter = Counter()
        self.ner_cnt: Counter = Counter()

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        self.min_frequency = configs.min_frequency
        self.normalize_digit = configs.normalize_digit
        self.embedding_path = configs.embedding_path

    def normalize_func(self, x):
        if self.normalize_digit:
            return normalize_digit_word(x)
        else:
            return x

    def _process(self, data_pack: DataPack):
        """
        Process the data pack to collect vocabulary information.

        Args:
            data_pack: The ner data to create vocabulary with.

        Returns:

        """
        # for data_pack in input_pack:
        for instance in data_pack.get_data(
                context_type=Sentence,
                request={Token: ["chunk", "pos", "ner"]}):
            for token in instance["Token"]["text"]:
                for char in token:
                    self.char_cnt[char] += 1
                word = self.normalize_func(token)
                self.word_cnt[word] += 1

            for pos in instance["Token"]["pos"]:
                self.pos_cnt[pos] += 1
            for chunk in instance["Token"]["chunk"]:
                self.chunk_cnt[chunk] += 1
            for ner in instance["Token"]["ner"]:
                self.ner_cnt[ner] += 1

    def finish(self, resource: Resources):
        # if a singleton is in pre-trained embedding dict,
        # set the count to min_occur + c
        for word in self.word_cnt:
            if self.word_cnt[word] < self.min_frequency:
                del self.word_cnt[word]

        word_alphabet = Alphabet("word", self.word_cnt)
        char_alphabet = Alphabet("character", self.char_cnt)
        ner_alphabet = Alphabet("ner", self.ner_cnt)

        embedding_dict = load_glove_embedding(self.embedding_path)

        for word in embedding_dict:
            if word not in word_alphabet.instance2index:
                word_alphabet.add(word)

        word_embedding_table = construct_word_embedding_table(embedding_dict,
                                                              word_alphabet)

        logging.info('word embedding table size: %s',
                     word_embedding_table.size())

        # Adding vocabulary information to resource.
        resource.update(
            word_alphabet=word_alphabet,
            char_alphabet=char_alphabet,
            ner_alphabet=ner_alphabet,
            word_embedding_table=word_embedding_table)
