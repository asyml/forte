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
"""A utility function for padding over dataser"""

import torch


def pad_each_bach(word, max_sen_len):
    batch_size = word.shape[0]
    curr_len = word.shape[1]
    word_list = word.tolist()

    # Line 0 in word_embedding_table is padding vec
    for i in range(batch_size):
        for j in range(max_sen_len - curr_len):
            word_list[i].append(0)

    return torch.LongTensor(word_list)
