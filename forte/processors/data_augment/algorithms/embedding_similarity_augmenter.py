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

import random
from typing import Dict, List
import numpy as np
from tensorflow import gfile
from texar.tf.data import Embedding, load_glove
from forte.processors.data_augment.algorithms.base_augmenter import BaseDataAugmenter

__all__ = [
    "EmbeddingSimilarityAugmenter",
]

random.seed(0)

emb_types = ['glove']


def load_glove_vocab(filename):
    vocab = {}
    idx_to_word = {}
    with gfile.GFile(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word = vec[0]
            word_idx = len(vocab)
            vocab[word] = word_idx
            idx_to_word[word_idx] = word
    return vocab, idx_to_word

def l2_norm(word_vecs):
    norm = np.sqrt((word_vecs * word_vecs).sum(axis=1))
    return word_vecs / norm[:, np.newaxis]

class EmbeddingSimilarityAugmenter(BaseDataAugmenter):
    r"""
    This class is a data augmenter leveraging pre-trained word
    embeddings, such as word2vec and glove, to replace the input
    word with another word with similar word embedding.
    By default, the replacement word is randomly chosen from the
    top k words with the most similar embeddings.

    Args:
        emb_path: Path to the pretrained embedding file
        emb_type: Type of embedding. E.g. glove
        emb_dim: Dimension of the embedding
    """
    def __init__(self, emb_path, emb_type, emb_dim, configs: Dict[str, str]):
        self.configs = configs
        self.emb_path = emb_path
        self.emb_type = emb_type
        self.top_k = int(self.configs["top_k"]) if "top_k" in self.configs else 100

        if emb_type == "glove":
            self.vocab, self.idx_to_word = load_glove_vocab(emb_path)
            hparams = Embedding.default_hparams()
            hparams["file"] = emb_path
            hparams["dim"] = emb_dim
            hparams["read_fn"] = load_glove
            embeddings = Embedding(self.vocab, hparams)
            self.normalized_vectors = l2_norm(embeddings.word_vecs)
        else:
            raise ValueError('Embedding type value is unexpected. \
                Expected values include {}'.format(emb_types))

    @property
    def replacement_level(self) -> List[str]:
        return ["word"]


    def augment(self, word: str, additional_info: Dict[str, str] = {}) -> str:
        r"""
        This function replaces a word with words with similar
        pretrained embeddings.
        Args:
            word: input
            additional_info: unused
        Returns:
            a replacement word with similar word embedding
        """
        if word not in self.vocab:
            return word

        source_id = self.vocab[word]
        source_vector = self.normalized_vectors[source_id]
        scores = np.dot(self.normalized_vectors, source_vector)
        target_ids = np.argpartition(-scores, self.top_k+1)[:self.top_k+1]
        target_words = [self.idx_to_word[idx] for idx in target_ids \
            if idx != source_id and self.idx_to_word[idx].lower() != word.lower()]
        return random.choice(target_words)
