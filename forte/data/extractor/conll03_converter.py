# Copyright 2020 The Forte Authors. All Rights Reserved.
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
The convert that handle conll03 data.
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
    "CoNLL03Converter"
]

class CoNLL03Converter:

    def __init__(self):
        self.token2id = {
            'U.N.': 0,
            'official': 1,
            'Ekeus': 2,
            'heads': 3,
            'for': 4,
            'Baghdad': 5,
            '.': 6
        }

        # Labels besides "O"
        self.labels = set(['I-ORG', 'I-PER', 'I-LOC'])
        self.positions = set(['B', 'I'])
        self.labelpos2id = {}
        curid = 0
        for l in self.labels:
            for p in self.positions:
                self.labelpos2id[(l, p)] = curid
                curid += 1

        # "O" label
        for p in self.positions:
            self.labelpos2id[('O', p)] = curid

    def convert(self, pack: DataPack):
        for sentence in pack.get(Sentence):
            token_ids = [self.token2id[token.text] for token
                            in pack.get(Token, sentence)]
            ners = [token.ner for token in pack.get(Token, sentence)]
            ners_ids = []
            if len(ners) == 0:
                continue
            elif len(ners) == 1:
                ners_ids.append(self.labelpos2id[(ners[0], 'B')])
            else:
                prev_label = None
                for cur_label in ners:
                    if cur_label == prev_label:
                        ners_ids.append(self.labelpos2id[(cur_label, 'I')])
                        prev_label = cur_label
                    else:
                        ners_ids.append(self.labelpos2id[(cur_label, 'B')])
                        prev_label = cur_label
            yield token_ids, ners_ids
