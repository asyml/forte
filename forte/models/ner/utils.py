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

import random
import re

import numpy as np
import torch

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(r"\d")


def normalize_digit_word(word):
    return DIGIT_RE.sub("0", word)


def load_glove_embedding(embedding_path, normalize_digits=True):
    """
    Load glove embeddings from file.

    Args:
        embedding_path:  the file to load embedding from.
        normalize_digits: whether to normalize the digits characters in token.

    Returns: embedding dict, embedding dimension, caseless

    """

    embed_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()
            if embed_dim < 0:
                embed_dim = len(tokens) - 1
            else:
                assert embed_dim + 1 == len(
                    tokens
                ), f"glove_dim{embed_dim} cur_dim{len(tokens)}"
            embedd = np.empty(embed_dim, dtype=np.float32)
            embedd[:] = tokens[1:]
            word = (
                DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
            )
            embedd_dict[word] = embedd

    return embedd_dict


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
