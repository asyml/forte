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
from texar.tf.data import Embedding
from texar.torch.data import Vocab

from forte.processors.data_augment.algorithms.base_augmenter import ReplacementDataAugmenter
from forte.processors.data_augment.utils.utils import l2_norm

__all__ = [
    "EmbeddingSimilarityAugmenter",
]

random.seed(0)

emb_types = ['glove']


class EmbeddingSimilarityAugmenter(ReplacementDataAugmenter):
    r"""
    This class is a data augmenter leveraging pre-trained word
    embeddings, such as word2vec and glove, to replace the input
    word with another word with similar word embedding.
    By default, the replacement word is randomly chosen from the
    top k words with the most similar embeddings.

    Args:
        embedding: A texar.tf.data.Embedding object. Can be initialized
            from pre-trained embedding file using helper functions 
            E.g. forte.processors.data_augment.utils.utils.load_glove_embedding
        vocab: A texar.torch.data.Vocab object. Can be initialized from
            pre-trained embedding file using helper functions
            E.g. forte.processor.data_augment.utils.utils.load_glove_vocab
        top_k: the number of k most similar words to choose from
    """
    def __init__(self, embedding: Embedding, vocab: Vocab, top_k: int = 100):
        super().__init__({})
        self.vocab = vocab
        self.top_k = top_k
        self.normalized_vectors = l2_norm(embedding.word_vecs)

    @property
    def replacement_level(self) -> List[str]:
        return ["word"]

    def augment(self, word: str) -> str:
        r"""
        This function replaces a word with words with similar
        pretrained embeddings.
        Args:
            word: input
            additional_info: unused
        Returns:
            a replacement word with similar word embedding
        """
        if word not in self.vocab.token_to_id_map_py:
            return word

        source_id = self.vocab.token_to_id_map_py[word]
        source_vector = self.normalized_vectors[source_id]
        scores = np.dot(self.normalized_vectors, source_vector)
        target_ids = np.argpartition(-scores, self.top_k+1)[:self.top_k+1]
        target_words = [self.vocab.id_to_token_map_py[idx] for idx in target_ids \
            if idx != source_id and self.vocab.id_to_token_map_py[idx].lower() != word.lower()]
        return random.choice(target_words)
