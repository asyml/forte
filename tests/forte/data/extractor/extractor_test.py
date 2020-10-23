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
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from ft.onto.base_ontology import Sentence, Token, Document, EntityMention
from forte.data.data_pack import DataPack
from forte.data.extractor.extractor import TextExtractor, CharExtractor, AnnotationSeqExtractor


class ExtractorTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03_new"

    def test_TextExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "scope": Sentence,
            "entry": Token,
            "conversion_method": "indexing"
        }
        extractor = TextExtractor(config)

        # Test token2id and labelpos2id dict
        tokens = ['U.N.', 'official', 'Ekeus', 'heads', 'for', 'Baghdad', '.']
        ners = ['ORG', 'O', 'PER', 'O', 'O', 'LOC', 'O']

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        tensors = []
        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                tensors.append(extractor.extract(pack, instance))

    def test_CharExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "scope": Sentence,
            "entry": Token,
            "vocab_method": "indexing",
            "vocab_use_pad": True
        }
        extractor = CharExtractor(config)

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        tensors = []
        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                tensors.append(extractor.extract(pack, instance))

    def test_AnnotationSeqExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "scope": Sentence,
            "entry": EntityMention,
            "attribute": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_method": "indexing"
        }

        extractor = AnnotationSeqExtractor(config)

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        tensors = []
        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                tensors.append(extractor.extract(pack, instance))

        expected = [[0, 3, 5, 3, 3, 7, 3]]
        tensors = [tensor.numpy() for tensor in tensors]
        tensors = [[int(x) for x in tensor] for tensor in tensors]
        self.assertListEqual(expected, tensors)

if __name__ == '__main__':
    unittest.main()
