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
Utility functions
"""

from os import path

import numpy as np
from texar.tf.data import Embedding, load_glove

from texar.torch.data import Vocab

__all__ = [
    "l2_norm",
    "load_glove_vocab"
]

def l2_norm(word_vecs):
    r"""Calculate the L2 norm

    Args:
        word_vecs: 2d numpy array of word vectors.

    Returns:
        Normalized vectors.
    """
    norm = np.sqrt((word_vecs * word_vecs).sum(axis=1))
    return word_vecs / norm[:, np.newaxis]

def load_glove_vocab(glove_filename: str) -> Vocab:
    r"""Load pre-trained glove vocabulary from file

    Args:
        glove_filename: Path to the pre-trained glove embeddings file

    Returns:
        A texar.tf.torch.data.Vocab object
    """
    vocab_filename = glove_filename + ".vocab"
    if not path.isfile(vocab_filename):
        vocab_list = []
        with open(glove_filename, "r") as glove_file:
            for line in glove_file:
                vec = line.strip().split()
                if len(vec) == 0:
                    continue
                word = vec[0]
                vocab_list.append(word)

        with open(vocab_filename, "w") as output_file:
            output_file.write('\n'.join(vocab_list))
    return Vocab(vocab_filename)

def load_glove_embedding(glove_filename: str, emb_dim: int, vocab: Vocab) -> Embedding:
    r"""Load pre-trained glove embeddings from file

    Args:
        glove_filename: Path to the pre-trained glove embeddings file
        emb_dim: Dimension of the embeddings
        vocab: A texar.torch.data.Vocab object

    Returns:
        A texar.tf.data.Embedding object
    """
    hparams = Embedding.default_hparams()
    hparams["file"] = glove_filename
    hparams["dim"] = emb_dim
    hparams["read_fn"] = load_glove
    tokan_to_id_map = vocab.token_to_id_map_py
    embedding = Embedding(vocab.token_to_id_map_py, hparams)
    return embedding
