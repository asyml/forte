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
"""This file predict the ner tag for conll03 dataset."""

import torch
from texar.torch.modules.embedders import WordEmbedder
from texar.torch.modules.classifiers.conv_classifiers import Conv1DClassifier
from torch import nn
from texar.torch.data import Batch

def pad_each_bach(word, max_sen_len):
    batch_size = word.shape[0]
    curr_len = word.shape[1]
    word_list = word.tolist()

    # Line 0 in word_embedding_table is padding vec
    for i in range(batch_size):
        for j in range(max_sen_len-curr_len):
            word_list[i].append(0)

    return torch.LongTensor(word_list)


class CNN_Classifier(nn.Module):
    def __init__(self, in_channels=None, word_embedding_table=None):
        super().__init__()
        self.embedder = WordEmbedder(init_value=word_embedding_table)

        self.classifier = \
            Conv1DClassifier(in_channels=in_channels,
                             in_features=word_embedding_table.size()[0])

        self.max_sen_len = in_channels

    def forward(self, batch: Batch):
        word = batch["text_tag"]["tensor"]

        word_pad = pad_each_bach(word, self.max_sen_len)

        word_pad_embed = self.embedder(word_pad)

        logits, pred = self.classifier(word_pad_embed)

        return logits, pred

