#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest

from typing import List, Tuple
from torch import Tensor
import torch

from data.converter.vocabulary import Vocabulary
from ft.onto.base_ontology import Sentence, Token, Document, EntityMention

from forte.data.data_pack import DataPack

from data.converter.converter import OneToOneConverter


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "John worked at Carnegie Mellon University",
            "Michael Jordan is a professor at Berkeley"
        ]

        self.data_pack: DataPack = self.build_data_pack(self.texts)

    def test_one_to_one_converter(self):
        config = {
            "scope": Sentence,
            "entry": Token,
            "repr": "text_repr",
            "conversion_method": "indexing"
        }
        converter = OneToOneConverter(config)

        for instance in self.data_pack.get_data(config["scope"]):
            converter.consume_instance(self.data_pack,
                                       instance[config["scope"]])

        tensors: List[Tensor] = converter.produce_instance()
        vocab: Vocabulary = converter.vocab

        self.assertEqual(len(tensors), 2)

        for tensor, text in zip(tensors, self.texts):
            for tensor_dim1, token in zip(tensor, text.split()):
                self.assertEqual(tensor_dim1,
                                 torch.tensor(vocab.to_id(token)))

        # TODO: test add_to_datapack

    def test_one_to_many_converter(self):
        config = {
            "scope": Sentence,
            "entry": Token,
            "repr": "char_repr",
            "conversion_method": "indexing"
        }
        converter = OneToManyConverter(config)

        for instance in self.data_pack.get_data(config["scope"]):
            converter.consume_instance(self.data_pack,
                                       instance[config["scope"]])

        tensors: List[Tensor] = converter.produce_instance()
        vocab: Vocabulary = converter.vocab

        self.assertEqual(len(tensors), 2)

        for tensor, text in zip(tensors, self.texts):
            self.assertEqual(len(tensor.shape), 2)
            for tensor_dim1, token in zip(tensor, text.split()):
                self.assertEqual(tensor_dim1.shape, (1, len(token)))
                for tensor_dim2, char in zip(tensor_dim1, iter(token)):
                    self.assertEqual(tensor_dim2,
                                 torch.tensor(vocab.to_id(char)))

        # TODO: test add_to_datapack

    def test_many_to_one_converter(self):
        config = {
            "scope": Sentence,
            "entry": EntityMention,
            "label": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "conversion_method": "indexing"
        }

        # Add NER into data pack
        ners = [
            [("PER", (0,3)), ("ORG", (15,40))],
            [("PER", (0,15)), ("LOC", (33,40))]
        ]
        self.add_ner(self.data_pack, ners)

        converter = ManyToOneConverter(config)

        for instance in self.data_pack.get_data(config["scope"]):
            converter.consume_instance(self.data_pack,
                                       instance[config["scope"]])

        tensors: List[Tensor] = converter.produce_instance()
        vocab: Vocabulary = converter.vocab

        self.assertEqual(len(tensors), 2)

        for tensor, ner in zip(tensors, ners):
            self.assertEqual(len(tensor.shape), 1)
            ner_idx = 0
            tag_o_id = vocab.to_id("O")
            ids = tensor.tolist()

            for curr_idx, curr_id in enumerate(ids):
                if curr_id != tag_o_id:
                    start_idx = curr_idx
                    if (curr_idx == len(ids)-1) or ids[curr_idx+1] == tag_o_id:
                        end_idx = curr_idx
                        ner_type = vocab.get_canonical_type(
                            vocab.from_id(ids[start_idx]))

                        expect_ner_type = ner[ner_idx][0]
                        expect_start_idx = ner[ner_idx][1][0]
                        expect_end_idx = ner[ner_idx][1][1]

                        self.assertEqual(ner_type, expect_ner_type)
                        self.assertEqual(start_idx, expect_start_idx)
                        self.assertEqual(end_idx, expect_end_idx)

                        ner_idx += 1
                    else:
                        curr_type = vocab.get_canonical_type(
                            vocab.from_id(ids[curr_idx]))
                        next_type = vocab.get_canonical_type(
                            vocab.from_id(ids[curr_idx+1]))

                        self.assertEqual(curr_type, next_type)

        # TODO: test add_to_datapack

    def test_many_to_many_converter(self):
        pass

    def build_data_pack(self, texts: List[str]) -> DataPack:
        data_pack: DataPack = DataPack()

        sentence_offset = 0
        entire_text = ""
        for text in texts:
            Sentence(data_pack, sentence_offset, len(text))
            sentence_offset += len(text) + 1

            token_offset = 0
            for token in text.split():
                Token(data_pack, token_offset, token_offset + len(token))
                token_offset += len(token) + 1

            if not entire_text:
                entire_text = " " + text

        Document(data_pack, 0, len(entire_text))
        data_pack.set_text(entire_text)

        return data_pack

    def add_ner(self, data_pack: DataPack, ners: List[List[Tuple]]):
        for ners_one_instance in ners:
            for ner in ners_one_instance:
                ner_type = ner[0]
                start_pos, end_pos = ner[1][0], ner[1][1]

                em = EntityMention(data_pack, start_pos, end_pos)
                em.ner_type = ner_type


if __name__ == '__main__':
    unittest.main()
