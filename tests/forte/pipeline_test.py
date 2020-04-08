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
Unit tests for Pipeline.
"""

import os
import unittest
from typing import Any, Dict, Iterator, Optional, Type

from ddt import ddt, data, unpack

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Generics
from forte.data.readers.base_reader import PackReader, MultiPackReader
from forte.data.selector import FirstPackSelector
from forte.pipeline import Pipeline
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor
from ft.onto.base_ontology import Token, Sentence
from tests.dummy_batch_processor import DummyRelationExtractor

data_samples_root = "data_samples"


class NewType(Generics):
    """A dummy generic type to check the correctness of pipeline execution."""

    def __init__(self, pack, value):
        super().__init__(pack)
        self.value = value


class SentenceReader(PackReader):
    """A simple sentence reader for pipeline tests."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def _collect(self, file_path) -> Iterator[Any]:  # type: ignore
        return iter([file_path])

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def text_replace_operation(self, text: str):
        return []

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        with open(file_path, "r", encoding="utf8") as doc:
            for line in doc:
                pack = DataPack(doc_id=file_path)
                line = line.strip()
                if len(line) == 0:
                    continue
                sent = Sentence(pack, 0, len(line))
                pack.add_entry(sent)
                pack.set_text(line)
                self.count += 1
                yield pack


class MultiPackSentenceReader(MultiPackReader):
    """A simple sentence reader for pipeline tests."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def _collect(self, file_path) -> Iterator[Any]:  # type: ignore
        return iter([file_path])

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def text_replace_operation(self, text: str):
        return []

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:  # type: ignore
        with open(file_path, "r", encoding="utf8") as doc:
            for line in doc:
                m_pack = MultiPack()
                pack = DataPack(doc_id=file_path)
                line = line.strip()
                if len(line) == 0:
                    continue
                sent = Sentence(pack, 0, len(line))
                pack.add_entry(sent)
                pack.set_text(line)
                self.count += 1
                m_pack.update_pack({"pack": pack})
                yield m_pack  # type: ignore


class DummyPackProcessor(PackProcessor):

    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        entries = list(input_pack.get_entries_by_type(NewType))
        if len(entries) == 0:
            entry = NewType(pack=input_pack, value="[PACK]")
            input_pack.add_entry(entry)
        else:
            entry = entries[0]  # type: ignore
            entry.value += "[PACK]"


class DummmyFixedSizeBatchProcessor(FixedSizeBatchProcessor):

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @staticmethod
    def _define_context() -> Type[Sentence]:
        return Sentence

    @staticmethod
    def _define_input_info() -> Dict:
        return {}

    def predict(self, data_batch: Dict):
        return data_batch

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        entries = list(data_pack.get_entries_by_type(NewType))
        if len(entries) == 0:
            entry = NewType(pack=data_pack, value="[BATCH]")
            data_pack.add_entry(entry)
        else:
            entry = entries[0]  # type: ignore
            entry.value += "[BATCH]"

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update(
            {
                "batcher": {"batch_size": 4}
            }
        )
        return config


@ddt
class PipelineTest(unittest.TestCase):

    def test_process_next(self):
        from forte.data.readers import OntonotesReader

        # Define and config the Pipeline
        nlp = Pipeline[DataPack]()
        nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = {"batcher": {"batch_size": 5}}
        nlp.add(dummy, config=config)
        nlp.initialize()

        dataset_path = os.path.join(data_samples_root, "ontonotes/00")

        # get processed pack from dataset
        for pack in nlp.process_dataset(dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                sent_text = sentence.text

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))

    def test_pipeline1(self):
        """Tests a pack processor only."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessor()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    def test_pipeline2(self):
        """Tests a batch processor only."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 4}}
        nlp.add(component=dummy, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(2, 4, 8)
    def test_pipeline3(self, batch_size):
        """Tests a chain of Batch->Pack->Batch with different batch sizes."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(4, 8, 16)
    def test_pipeline4(self, batch_size):
        """Tests a chain of Pack->Batch->Pack."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add(component=dummy1)

        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy2, config=config)

        dummy3 = DummyPackProcessor()
        nlp.add(component=dummy3)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3), (4, 5), (8, 9), (3, 2), (5, 4), (9, 8))
    @unpack
    def test_pipeline5(self, batch_size1, batch_size2):
        # Tests a chain of Batch->Pack->Batch with different batch sizes.

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline6(self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch with different batch sizes.

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline7(self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch->Pack with different batch sizes.

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config)
        dummy4 = DummyPackProcessor()
        nlp.add(component=dummy4)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)


@ddt
class MultiPackPipelineTest(unittest.TestCase):

    def test_process_next(self):
        from forte.data.readers import OntonotesReader

        # Define and config the Pipeline
        nlp = Pipeline[DataPack]()
        nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = {"batcher": {"batch_size": 5}}
        nlp.add(dummy, config=config)
        nlp.initialize()

        dataset_path = data_samples_root + "/ontonotes/00"

        # get processed pack from dataset
        for pack in nlp.process_dataset(dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                sent_text = sentence.text

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))

    def test_pipeline1(self):
        """Tests a pack processor only."""

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessor()
        nlp.add(dummy, selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    def test_pipeline2(self):
        """Tests a batch processor only."""

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 4}}
        nlp.add(component=dummy, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(2, 4, 8)
    def test_pipeline3(self, batch_size):
        """Tests a chain of Batch->Pack->Batch with different batch sizes."""
        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2, selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(4, 8, 16)
    def test_pipeline4(self, batch_size):
        """Tests a chain of Pack->Batch->Pack."""

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add(component=dummy1, selector=FirstPackSelector())

        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())

        dummy3 = DummyPackProcessor()
        nlp.add(component=dummy3,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3), (4, 5), (8, 9), (3, 2), (5, 4), (9, 8))
    @unpack
    def test_pipeline5(self, batch_size1, batch_size2):
        # Tests a chain of Batch->Pack->Batch with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2,
                selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline6(self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline7(self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch->Pack with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        dummy4 = DummyPackProcessor()
        nlp.add(component=dummy4, selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_by_type(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)


if __name__ == '__main__':
    unittest.main()
