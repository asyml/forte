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

"""
Unit tests for the writers.
"""
import os
import shutil
import tempfile
import unittest
from typing import List, Dict

from forte.data.data_pack import DataPack
from forte.data.readers import OntonotesReader, \
    RecursiveDirectoryDeserializeReader
from forte.pipeline import Pipeline
from forte.processors.annotation_remover import AnnotationRemover
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter
from forte.processors.writers import PackNameJsonPackWriter
from ft.onto.base_ontology import Token


class TestJsonWriter(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '..', '..', '..',
                'data_samples', 'ontonotes/00')
        )

    def test_serialize_deserialize_processor(self):
        pipe_serialize = Pipeline[DataPack]()
        pipe_serialize.set_reader(OntonotesReader())
        pipe_serialize.add(
            AnnotationRemover(),
            # Remove tokens and sentences form OntonotesReader.
            {'removal_types':
                [
                    'ft.onto.base_ontology.Token',
                    'ft.onto.base_ontology.Sentence',
                ]}
        )
        pipe_serialize.add(NLTKSentenceSegmenter())
        pipe_serialize.add(NLTKWordTokenizer())
        pipe_serialize.add(NLTKPOSTagger())

        with tempfile.TemporaryDirectory() as output_dir:
            pipe_serialize.add(
                PackNameJsonPackWriter(), {
                    'output_dir': output_dir,
                    'indent': 2,
                }
            )

            pipe_serialize.run(self.data_path)

            pipe_deserialize = Pipeline[DataPack]()
            pipe_deserialize.set_reader(RecursiveDirectoryDeserializeReader())
            pipe_deserialize.initialize()

            token_counts: Dict[str, int] = {}

            # This basically test whether the deserialized data is
            # still the same as expected.
            pack: DataPack
            for pack in pipe_deserialize.process_dataset(output_dir):
                tokens: List[Token] = list(pack.get(Token))
                token_counts[pack.pack_name] = len(tokens)

            expected_count = {
                'bn/abc/00/abc_0039': 72, 'bn/abc/00/abc_0019': 370,
                'bn/abc/00/abc_0059': 39, 'bn/abc/00/abc_0009': 424,
                'bn/abc/00/abc_0029': 487, 'bn/abc/00/abc_0069': 428,
                'bn/abc/00/abc_0049': 73}

            assert token_counts == expected_count
