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

from ddt import ddt, data, unpack
from texar.torch.data import Embedding, load_glove

from forte.data.caster import MultiPackBoxer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.processors.data_augment.data_aug_processor import (
    DataAugProcessor,
)
from forte.processors.data_augment.algorithms.embedding_similarity_replacement_op import (
    EmbeddingSimilarityReplacementOp,
)
from forte.processors.misc import WhiteSpaceTokenizer
from ft.onto.base_ontology import Token


@ddt
class TestEmbeddingSimilarityReplacementOp(unittest.TestCase):
    def setUp(self):
        file_dir_path = os.path.dirname(__file__)
        vocab_path = (
            "tests/forte/processors/data_augment/algorithms/"
            "sample_embedding.txt.vocab"
        )
        self.abs_vocab_path = os.path.abspath(
            os.path.join(file_dir_path, *([os.pardir] * 5), vocab_path)
        )
        embed_path = (
            "tests/forte/processors/data_augment/algorithms/"
            "sample_embedding.txt"
        )
        abs_embed_path = os.path.abspath(
            os.path.join(file_dir_path, *([os.pardir] * 5), embed_path)
        )
        embed_hparams = Embedding.default_hparams()
        embed_hparams["file"] = abs_embed_path
        embed_hparams["dim"] = 50
        embed_hparams["read_fn"] = load_glove
        self.embed_hparams = embed_hparams
        self.esa = EmbeddingSimilarityReplacementOp(
            configs={
                "vocab_path": self.abs_vocab_path,
                "embed_hparams": self.embed_hparams,
                "top_k": 5,
            }
        )

    def test_replace(self):
        data_pack = DataPack()
        data_pack.set_text("google")
        token_1 = Token(data_pack, 0, 6)
        data_pack.add_entry(token_1)
        augmented_data_pack = self.esa.perform_augmentation(data_pack)
        augmented_token = list(
            augmented_data_pack.get("ft.onto.base_ontology.Token")
        )[0]
        self.assertIn(
            augmented_token.text,
            ["yahoo", "aol", "microsoft", "web", "internet"],
        )

    @data(
        (
            ["he google yahoo"],
            ["his yahoo aol"],
        )
    )
    @unpack
    def test_pipeline(self, texts, expected_outputs):
        nlp = Pipeline[MultiPack]()

        boxer_config = {"pack_name": "input"}

        nlp.set_reader(reader=StringReader())
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())

        processor_config = {
            "data_aug_op": "forte.processors.data_augment.algorithms"
            ".embedding_similarity_replacement_op."
            "EmbeddingSimilarityReplacementOp",
            "data_aug_op_config": {
                "other_entry_policy": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align",
                },
                "augment_entry": "ft.onto.base_ontology.Token",
                "vocab_path": self.abs_vocab_path,
                "embed_hparams": self.embed_hparams,
                "top_k": 1,
            },
            "augment_pack_names": {"input": "augmented_input"},
        }
        nlp.add(component=(DataAugProcessor()), config=processor_config)
        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(texts)):
            aug_pack = m_pack.get_pack("augmented_input")
            self.assertEqual(aug_pack.text, expected_outputs[idx])


if __name__ == "__main__":
    unittest.main()
