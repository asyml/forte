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
from typing import Tuple, Dict, Any
import numpy as np

from ft.onto.base_ontology import Annotation
from forte.utils import create_import_error_msg
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.single_annotation_op import (
    SingleAnnotationAugmentOp,
)

try:
    from texar.torch.data import (
        Vocab,
        Embedding,
    )
except ImportError as e:
    raise ImportError(
        create_import_error_msg(
            "texar-pytorch",
            "data_aug",
            "EmbeddingSimilarityReplacementOp",
        )
    ) from e

__all__ = [
    "EmbeddingSimilarityReplacementOp",
]


class EmbeddingSimilarityReplacementOp(SingleAnnotationAugmentOp):
    r"""
    This class is a replacement op leveraging pre-trained word
    embeddings, such as `word2vec` and `glove`, to replace the input
    word with another word with similar word embedding.
    By default, the replacement word is randomly chosen from the
    top k words with the most similar embeddings.
    """

    def __init__(self, configs: Config):
        super().__init__(configs)
        self.vocab = Vocab(self.configs["vocab_path"])
        embed_hparams = self.configs["embed_hparams"]
        embedding = Embedding(self.vocab.token_to_id_map_py, embed_hparams)
        self.normalized_vectors = (
            embedding.word_vecs
            / np.sqrt((embedding.word_vecs**2).sum(axis=1))[:, np.newaxis]
        )

    def single_annotation_augment(
        self, input_anno: Annotation
    ) -> Tuple[bool, str]:
        r"""
        This function replaces a word words with similar
        pretrained embeddings.

        Args:
            input_anno (Annotation): The input annotation.
        Returns:
            A tuple of two values, where the first element is a boolean value
            indicating whether the replacement happens, and the second
            element is the replaced word.
        """
        word = input_anno.text
        if word not in self.vocab.token_to_id_map_py:
            return False, word

        source_id = self.vocab.token_to_id_map_py[word]
        source_vector = self.normalized_vectors[source_id]
        scores = np.dot(self.normalized_vectors, source_vector)
        target_ids = np.argpartition(-scores, self.configs["top_k"] + 1)[
            : self.configs["top_k"] + 1
        ]
        target_words = [
            self.vocab.id_to_token_map_py[idx]
            for idx in target_ids
            if idx != source_id
            and self.vocab.id_to_token_map_py[idx].lower() != word.lower()
        ]
        return True, random.choice(target_words)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""
        returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            `"vocab_path"`: str
                The absolute path to the vocabulary file for
                the pretrained embeddings

            `"embed_hparams"`: dict
                The hyper-parameters to initialize the
                texar.torch.data.Embedding object.

            `"top_k"`: int
                the number of k most similar words to choose from
        """
        return {
            "vocab_path": "",
            "embed_hparams": Embedding.default_hparams(),
            "top_k": 0,
        }
