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
Unit tests for dictionary word replacement data augmenter.
"""
import os
import unittest

from forte.processors.data_augment.algorithms.embedding_similarity_replacement_op \
    import EmbeddingSimilarityReplacementOp
from forte.processors.data_augment.utils.utils import load_glove_vocab, load_glove_embedding

from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack


class TestEmbeddingSimilarityReplacementOp(unittest.TestCase):
    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        vocab_path = "tests/forte/processors/data_augment/algorithms/"\
                     "sample_embedding.txt"
        abs_vocab_path = os.path.abspath(os.path.join(file_dir_path,
                                                      *([os.pardir] * 5),
                                                      vocab_path))
        self.vocab = load_glove_vocab(abs_vocab_path)
        self.embedding = load_glove_embedding(abs_vocab_path, 50, self.vocab)
        self.esa = EmbeddingSimilarityReplacementOp(
            self.embedding,
            self.vocab,
            configs={
                "top_k": 5
            }
        )

    def test_replace(self):
        data_pack = DataPack()
        data_pack.set_text("google")
        token_1 = Token(data_pack, 0, 6)
        data_pack.add_entry(token_1)
        self.assertIn(
            self.esa.replace(token_1),
            ['yahoo', 'aol', 'microsoft', 'web', 'internet']
        )


if __name__ == "__main__":
    unittest.main()
