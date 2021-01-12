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

from texar.torch.modules.embedders import WordEmbedder
from texar.torch.modules.classifiers.conv_classifiers import Conv1DClassifier
from torch import nn
from texar.torch.data import Batch
from examples.classification_example.util import pad_each_bach


class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, word_embedding_table):
        super().__init__()
        self.embedder = WordEmbedder(init_value=word_embedding_table)

        self.classifier = \
            Conv1DClassifier(in_channels=in_channels,
                             in_features=word_embedding_table.size()[1])

        self.max_sen_len = in_channels

    def forward(self, batch: Batch):
        word = batch["text_tag"]["data"]

        word_pad = pad_each_bach(word, self.max_sen_len)

        word_pad_embed = self.embedder(word_pad)

        logits, pred = self.classifier(word_pad_embed)

        return logits, pred
