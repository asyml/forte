"""
Unit tests for Pipeline.
"""
# pylint: disable=no-self-use,unused-argument,useless-super-delegation
import unittest
from ddt import ddt, data, unpack
import os
from typing import Any, Dict, Iterator, Optional, Type

from texar.torch import HParams

from forte.common import Resources
from forte.data.readers import PackReader, MultiPackReader, OntonotesReader
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Generic
from forte.data.selector import FirstPackSelector
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor
from forte.processors.base.tests.dummy_batch_processor import \
    DummyRelationExtractor
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence, RelationLink


class NewType(Generic):
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

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
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
                yield m_pack


class DummyPackProcessor(PackProcessor):

    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        entries = list(input_pack.get_entries_by_type(NewType))
        if len(entries) == 0:
            entry = NewType(pack=input_pack, value="[PACK]")
            input_pack.add_entry(entry)
        else:
            entry = entries[0]
            entry.value += "[PACK]"


class DummmyFixedSizeBatchProcessor(FixedSizeBatchProcessor):

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
        self.batcher = self.define_batcher()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        self.batcher.initialize(configs.batcher)

    def define_context(self) -> Type[Sentence]:
        return Sentence

    def _define_input_info(self) -> Dict:
        return {}

    def predict(self, data_batch: Dict):
        return data_batch

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        entries = list(data_pack.get_entries_by_type(NewType))
        if len(entries) == 0:
            entry = NewType(pack=data_pack, value="[BATCH]")
            data_pack.add_entry(entry)
        else:
            entry = entries[0]
            entry.value += "[BATCH]"

    @staticmethod
    def default_hparams():
        return {
            "batcher": {"batch_size": 4}
        }


@ddt
class PipelineTest(unittest.TestCase):

    def test_process_next(self):

        # Define and config the Pipeline
        nlp = Pipeline()
        nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = HParams({"batcher": {"batch_size": 5}},
                         dummy.default_hparams())
        nlp.add_processor(dummy, config=config)
        nlp.initialize()

        dataset_path = \
            "forte/tests/data_samples/ontonotes_sample_dataset/00"

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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessor()
        nlp.add_processor(dummy)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"
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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": 4}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy, config=config)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"
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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add_processor(processor=dummy2)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": 2 * batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add_processor(processor=dummy1)

        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config)

        dummy3 = DummyPackProcessor()
        nlp.add_processor(processor=dummy3, config=config)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add_processor(processor=dummy2)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config)
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size3}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config)
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config)
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size3}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config)
        dummy4 = DummyPackProcessor()
        nlp.add_processor(processor=dummy4)
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        # Define and config the Pipeline
        nlp = Pipeline()
        nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = HParams({"batcher": {"batch_size": 5}},
                         dummy.default_hparams())
        nlp.add_processor(dummy, config=config)
        nlp.initialize()

        dataset_path = \
            "forte/tests/data_samples/ontonotes_sample_dataset/00"

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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessor()
        nlp.add_processor(dummy, selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"
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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": 4}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy, config=config,
                          selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"
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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config,
                          selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add_processor(processor=dummy2, selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": 2 * batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config,
                          selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add_processor(processor=dummy1, selector=FirstPackSelector())

        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config,
                          selector=FirstPackSelector())

        dummy3 = DummyPackProcessor()
        nlp.add_processor(processor=dummy3, config=config,
                          selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config,
                          selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add_processor(processor=dummy2,
                          selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config,
                          selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config,
                          selector=FirstPackSelector())
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config,
                          selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size3}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config,
                          selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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

        nlp = Pipeline()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size1}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy1, config=config,
                          selector=FirstPackSelector())
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size2}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy2, config=config,
                          selector=FirstPackSelector())
        dummy3 = DummmyFixedSizeBatchProcessor()
        config = HParams({"batcher": {"batch_size": batch_size3}},
                         DummmyFixedSizeBatchProcessor.default_hparams())
        nlp.add_processor(processor=dummy3, config=config,
                          selector=FirstPackSelector())
        dummy4 = DummyPackProcessor()
        nlp.add_processor(processor=dummy4, selector=FirstPackSelector())
        nlp.initialize()
        data_path = "forte/processors/base/tests/data_samples/" \
                    "random_texts/0.txt"

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
