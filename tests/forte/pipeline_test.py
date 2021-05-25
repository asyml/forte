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
import re
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Type, Set

import numpy as np
from ddt import ddt, data, unpack

from forte.data.base_pack import PackType
from forte.data.base_reader import PackReader, MultiPackReader
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.caster import MultiPackBoxer
from forte.data.converter import Converter
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Generics
from forte.data.readers import PlainTextReader, StringReader, OntonotesReader
from forte.data.selector import FirstPackSelector, NameMatchSelector, \
    SinglePackSelector, AllPackSelector
from forte.data.types import DataRequest
from forte.common import ProcessExecutionException, ProcessorConfigError, \
    Resources
from forte.evaluation.base import Evaluator
from forte.pipeline import Pipeline
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor, \
    MultiPackProcessor
from forte.processors.base.batch_processor import Predictor, BatchProcessor
from forte.train_preprocessor import TrainPreprocessor
from forte.utils import get_full_module_name
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink


data_samples_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    *([os.path.pardir] * 2), 'data_samples'))

onto_specs_samples_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    *([os.path.pardir] * 1), 'forte', 'data', 'ontology', 'test_specs'))


@dataclass
class NewType(Generics):
    """A dummy generic type to check the correctness of pipeline execution."""
    value: Optional[str] = None

    def __init__(self, pack, value):
        super().__init__(pack)
        self.value = value


class NothingSelector(SinglePackSelector):
    """Select no pack from the :class:`MultiPack`"""

    def select(self, m_pack: MultiPack) -> Iterator[DataPack]:
        yield from []


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
                pack = DataPack(file_path)
                line = line.strip()
                if len(line) == 0:
                    continue

                pack.set_text(line)
                Sentence(pack, 0, len(line))
                self.count += 1

                yield pack


class MultiPackSentenceReader(MultiPackReader):
    """A simple sentence reader for pipeline tests. This creates a multipack
    with only one pack inside."""

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
                line = line.strip()
                if len(line) == 0:
                    continue

                m_pack = MultiPack()
                pack = m_pack.add_pack('pack')
                pack.set_text(line)

                Sentence(pack, 0, len(line))
                self.count += 1

                yield m_pack  # type: ignore


class MultiPackCopier(MultiPackProcessor):
    """
    Create a new pack inside the multi pack, make a copy of the first pack.
    """

    def _process(self, input_pack: MultiPack):
        pack = input_pack.add_pack('copy')
        pack.set_text(input_pack.get_pack_at(0).text)


class PeriodSentenceSplitter(PackProcessor):
    def _process(self, input_pack: DataPack):
        pattern = '\\.\\s*'
        start = 0

        for m in re.finditer(pattern, input_pack.text):
            end = m.end()
            Sentence(input_pack, start, end)
            start = end


class DummyRelationExtractor(BatchProcessor):
    r"""A dummy relation extractor.

    Note that to use :class:`DummyRelationExtractor`, the :attr:`ontology` of
    :class:`Pipeline` must be an ontology that includes
    ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def define_batcher() -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()

    @staticmethod
    def _define_context() -> Type[Sentence]:
        return Sentence

    @staticmethod
    def _define_input_info() -> DataRequest:
        input_info: DataRequest = {
            Token: [],
            EntityMention: {"fields": ["ner_type", "tid"]}
        }
        return input_info

    def predict(self, data_batch: Dict):
        entities_span = data_batch["EntityMention"]["span"]
        entities_tid = data_batch["EntityMention"]["tid"]

        pred: Dict = {
            "RelationLink": {
                "parent.tid": [],
                "child.tid": [],
                "rel_type": [],
            }
        }
        for tid, entity in zip(entities_tid, entities_span):
            parent = []
            child = []
            rel_type = []

            entity_num = len(entity)
            for i in range(entity_num):
                for j in range(i + 1, entity_num):
                    parent.append(tid[i])
                    child.append(tid[j])
                    rel_type.append("dummy_relation")

            pred["RelationLink"]["parent.tid"].append(
                np.array(parent))
            pred["RelationLink"]["child.tid"].append(
                np.array(child))
            pred["RelationLink"]["rel_type"].append(
                np.array(rel_type))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        r"""Add corresponding fields to data_pack"""
        if output_dict is None:
            return

        for i in range(len(output_dict["RelationLink"]["parent.tid"])):
            for j in range(len(output_dict["RelationLink"]["parent.tid"][i])):
                link = RelationLink(data_pack)
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["parent.tid"][i][j])
                link.set_parent(parent)
                child: EntityMention = data_pack.get_entry(  # type: ignore
                    output_dict["RelationLink"]["child.tid"][i][j])
                link.set_child(child)

    @classmethod
    def default_configs(cls):
        configs = super().default_configs()
        configs["batcher"] = {"batch_size": 10}
        return configs


class DummyEvaluator(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pass

    def get_result(self) -> Any:
        pass


class DummyPackProcessor(PackProcessor):

    def __init__(self):
        super().__init__()
        # Use to test the initialization behavior.
        self.initialize_count = 0

    def initialize(self, resources, configs):
        super().initialize(resources, configs)
        if ("successor" in configs["test"] and "test" not in configs["test"]):
                raise ProcessorConfigError('"test" is necessary as the first '
                                           'step for "successor" in config '
                                           'for test case purpose.')
        self.initialize_count += 1

    def _process(self, input_pack: DataPack):
        entries = list(input_pack.get_entries_of(NewType))
        if len(entries) == 0:
            NewType(pack=input_pack, value="[PACK]")
        else:
            entry = entries[0]  # type: ignore
            entry.value += "[PACK]"

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        configs['test'] = "test, successor"
        return configs


class DummyFixedSizeBatchProcessor(FixedSizeBatchProcessor):

    def __init__(self):
        super().__init__()
        self.counter = 0

    @staticmethod
    def _define_context() -> Type[Sentence]:
        return Sentence

    @staticmethod
    def _define_input_info() -> Dict:
        return {}

    def predict(self, data_batch: Dict):
        self.counter += 1
        return data_batch

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        entries = list(data_pack.get_entries_of(NewType))
        if len(entries) == 0:
            NewType(pack=data_pack, value="[BATCH]")
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


class DummyModel:
    """Dummy Model."""

    def __call__(self, batch):
        """Dummy model does nothing."""
        pass


class DummyPredictor(Predictor):
    """Dummy Predictor."""

    def predict(self, batch):
        return {}


@ddt
class PredictorPipelineTest(unittest.TestCase):

    @data(2, 4, 8)
    def test_pipeline_different_batch_size_chain_predictor(self, batch_size):
        """Tests a chain of Batch->Pack->Batch with different batch sizes."""

        data_path = data_samples_root + "/random_texts/0.txt"
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(SentenceReader())
        pipeline.initialize()

        text_extractor_name = "forte.data.extractors.AttributeExtractor"
        text_extractor_config = {
            "need_pad": True,
            "entry_type": "ft.onto.base_ontology.Token",
            "attribute": "text",
        }

        model = DummyModel()
        predictor = DummyPredictor()
        predictor_config = {
            "scope": "ft.onto.base_ontology.Sentence",
            "batch_size": batch_size,
            "feature_scheme": {
                "text_tag": {
                    "extractor": {
                        "class_name": text_extractor_name,
                        "config": text_extractor_config
                    },
                    "type": "data_input"
                },
            },
        }
        predictor.load(model)

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        nlp.add(predictor, config=predictor_config)
        nlp.add(DummyEvaluator())
        nlp.initialize()

        text_extractor = predictor.configs.\
            feature_scheme.text_tag.extractor
        for pack in pipeline.process_dataset(data_path):
            for instance in pack.get(Sentence):
                text_extractor.update_vocab(pack, instance)

        num_packs = 0
        for _ in nlp.process_dataset(data_path):
            num_packs += 1

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)


@ddt
class PipelineTest(unittest.TestCase):

    def test_process_next(self):
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
            for sentence in pack.get(Sentence):
                sent_text = sentence.text

                # second method to get entry in a sentence
                tokens = [token.text for token in pack.get(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))

    def test_pipeline_invalid_config(self):
        # Test a invalid config
        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessor()
        config = {'test': 'successor'}
        nlp.add(dummy, config=config)
        print(nlp.components)

        with self.assertRaises(ProcessorConfigError):
            nlp.initialize()


    def test_pipeline_pack_processor(self):
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
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    def test_pipeline_batch_processor(self):
        """Tests a batch processor only."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 4}}
        nlp.add(component=dummy, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(2, 4, 8)
    def test_pipeline_different_batch_size_chain(self, batch_size):
        """Tests a chain of Batch->Pack->Batch with different batch sizes."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2)
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(4, 8, 16)
    def test_pipeline_pack_batch_pack_chain(self, batch_size):
        """Tests a chain of Pack->Batch->Pack."""

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add(component=dummy1)

        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy2, config=config)

        dummy3 = DummyPackProcessor()
        nlp.add(component=dummy3)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3), (4, 5), (8, 9), (3, 2), (5, 4), (9, 8))
    @unpack
    def test_pipeline_batch_pack_batch_diff_size(self, batch_size1,
                                                 batch_size2):
        # Tests a chain of Batch->Pack->Batch with different batch sizes.
        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2)
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline_three_stack_batch_diff_size(self,
                                                  batch_size1,
                                                  batch_size2,
                                                  batch_size3):
        # Tests a chain of Batch->Batch->Batch with different batch sizes.

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config)
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline_three_stack_diff_size_batch_pack_chain(self,
                                                             batch_size1,
                                                             batch_size2,
                                                             batch_size3):
        # Tests a chain of Batch->Batch->Batch->Pack with different batch sizes.

        nlp = Pipeline[DataPack]()
        reader = SentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config)
        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config)
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config)
        dummy4 = DummyPackProcessor()
        nlp.add(component=dummy4)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)


@ddt
class MultiPackPipelineTest(unittest.TestCase):
    def test_process_multi_next(self):
        from forte.data.readers import OntonotesReader

        # Define and config the Pipeline
        nlp = Pipeline[DataPack]()
        nlp.set_reader(OntonotesReader())

        pack_name = 'test_pack'
        nlp.add(MultiPackBoxer(), {'pack_name': pack_name})
        nlp.add(
            DummyRelationExtractor(),
            config={"batcher": {"batch_size": 5}},
            selector=NameMatchSelector(select_name=pack_name)
        )
        nlp.initialize()

        dataset_path = data_samples_root + "/ontonotes/00"

        # get processed pack from dataset
        m_pack: MultiPack
        for m_pack in nlp.process_dataset(dataset_path):
            pack = m_pack.get_pack(pack_name)
            # get sentence from pack
            for sentence in pack.get(Sentence):
                sent_text = sentence.text

                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))

    def test_pipeline_multipack_reader(self):
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
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    def test_pipeline_multipack_selector(self):
        """Tests a batch processor only."""

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 4}}
        nlp.add(component=dummy, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(1, 2, 3)
    def test_one_batch_processor(self, batch_size):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(StringReader())
        batch_processor = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(PeriodSentenceSplitter())
        nlp.add(batch_processor, config=config)
        nlp.initialize()
        sentences = ["This tool is called Forte. The goal of this project to "
                     "help you build NLP pipelines. NLP has never been made "
                     "this easy before."]
        pack = nlp.process(sentences)
        sent_len = len(list(pack.get(Sentence)))
        self.assertEqual(
            batch_processor.counter,
            (sent_len // batch_size + (sent_len % batch_size > 0)))

    @data(1, 2, 3)
    def test_two_batch_processors(self, batch_size):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(PlainTextReader())
        dummy1 = DummyFixedSizeBatchProcessor()
        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(PeriodSentenceSplitter())

        nlp.add(dummy1, config=config)
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add(dummy2, config=config)

        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts")
        pack = nlp.process(data_path)
        sent_len = len(list(pack.get(Sentence)))

        self.assertEqual(
            dummy1.counter, (sent_len // batch_size +
                             (sent_len % batch_size > 0)))

        self.assertEqual(
            dummy2.counter, (sent_len // (2 * batch_size) +
                             (sent_len % (2 * batch_size) > 0)))

    @data(2, 4, 8)
    def test_pipeline_multipack_batch_pack_batch_double_size(self, batch_size):
        """Tests a chain of Batch->Pack->Batch with different batch sizes."""
        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2, selector=FirstPackSelector())
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts", "0.txt")

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data(4, 8, 16)
    def test_pipeline_multipack_pack_batch_pack_chain(self, batch_size):
        """Tests a chain of Pack->Batch->Pack."""

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyPackProcessor()
        nlp.add(component=dummy1, selector=FirstPackSelector())

        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())

        dummy3 = DummyPackProcessor()
        nlp.add(component=dummy3,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts", "0.txt")

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[PACK][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3), (4, 5), (8, 9), (3, 2), (5, 4), (9, 8))
    @unpack
    def test_pipeline_multipack_batch_pack_batch_diff_size(self,
                                                           batch_size1,
                                                           batch_size2):
        # Tests a chain of Batch->Pack->Batch with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyPackProcessor()
        nlp.add(component=dummy2,
                selector=FirstPackSelector())
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts", "0.txt")

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][PACK][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline_multipack_three_stack_batch_diff(
            self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts", "0.txt")

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    @data((2, 3, 4), (4, 5, 3), (8, 9, 7))
    @unpack
    def test_pipeline_multipack_three_stack_batch_diff_size_pack_chain(
            self, batch_size1, batch_size2, batch_size3):
        # Tests a chain of Batch->Batch->Batch->Pack with different batch sizes.

        nlp = Pipeline[MultiPack]()
        reader = MultiPackSentenceReader()
        nlp.set_reader(reader)
        dummy1 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size1}}
        nlp.add(component=dummy1, config=config,
                selector=FirstPackSelector())
        dummy2 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size2}}
        nlp.add(component=dummy2, config=config,
                selector=FirstPackSelector())
        dummy3 = DummyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size3}}
        nlp.add(component=dummy3, config=config,
                selector=FirstPackSelector())
        dummy4 = DummyPackProcessor()
        nlp.add(component=dummy4, selector=FirstPackSelector())
        nlp.initialize()
        data_path = os.path.join(data_samples_root, "random_texts", "0.txt")

        num_packs = 0
        for pack in nlp.process_dataset(data_path):
            types = list(pack.get_pack("pack").get_entries_of(NewType))
            num_packs += 1
            self.assertEqual(len(types), 1)
            self.assertEqual(types[0].value, "[BATCH][BATCH][BATCH][PACK]")

        # check that all packs are yielded
        self.assertEqual(num_packs, reader.count)

    def test_empty_selector(self):
        """
        Test the selector that doesn't select anything perform well in the
        pipeline.
        """
        for pack in Pipeline().set_reader(MultiPackSentenceReader()).add(
                DummyPackProcessor(), selector=NothingSelector()
        ).initialize().process_dataset(
            os.path.join(data_samples_root, "random_texts", "0.txt")
        ):
            # Because no packs are selected, we do not have any entries added.
            self.assertTrue(pack.get_pack('pack').num_generics_entries == 0)

    def test_caster_all_selector(self):
        """
        Test if the caster and all pack selector works well.
        The caster is used to convert a single pack to multi pack, and then
        pack copier is used to create a new pack. The all pack selector selects
        all the pack from the multi pack. This test make sure this pipeline
        works OK.
        """
        mp: MultiPack
        for mp in Pipeline().set_reader(SentenceReader()).add(
                MultiPackBoxer()
        ).add(
            MultiPackCopier()
        ).add(
            DummyPackProcessor(), selector=AllPackSelector()
        ).initialize().process_dataset(
            os.path.join(data_samples_root, "random_texts", "0.txt")
        ):
            num_pack = 0
            for pack in mp.packs:
                num_pack += 1
                entries = list(pack.get(NewType))
                self.assertEqual(len(entries), 1)
                self.assertEqual(
                    entries[0].value, "[PACK]")
            self.assertEqual(num_pack, 2)


class DummySentenceReaderOne(SentenceReader):

    def record(self, record_meta: Dict[str, Set[str]]):
        record_meta["Sentence"] = {"1", "2", "3"}


class DummySentenceReaderTwo(SentenceReader):

    def record(self, record_meta: Dict[str, Set[str]]):
        record_meta["ft.onto.example_ontology.Word"] = {"string_features",
                                                        "word_forms",
                                                        "token_ranks"}


class DummyPackProcessorOne(DummyPackProcessor):

    def record(self, record_meta: Dict[str, Set[str]]):
        record_meta["Token"] = {"1", "2"}
        record_meta["Document"] = {"2"}

    @classmethod
    def expected_types_and_attributes(cls):
        expectation: Dict[str, Set[str]] = {
            "Sentence": {"1", "2", "3"}
        }

        return expectation


class DummyPackProcessorTwo(DummyPackProcessor):

    def record(self, record_meta: Dict[str, Set[str]]):
        record_meta["Token"] = {"1", "2"}
        record_meta["Document"] = {"2"}

    @classmethod
    def expected_types_and_attributes(cls):
        expectation: Dict[str, Set[str]] = {
            "Document": {"1", "2", "3", "4"}
        }

        return expectation


class DummyPackProcessorThree(DummyPackProcessor):

    @classmethod
    def expected_types_and_attributes(cls):
        expectation: Dict[str, Set[str]] = {
            "ft.onto.example_import_ontology.Token": {"pos", "lemma"}
        }

        return expectation


class DummyEvaluatorOne(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def pred_pack_record(self, record_meta: Dict[str, Set[str]]):
        record_meta["Token"] = {"1", "2"}

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_pack_expectation: Dict[str, Set[str]] = {
            "Sentence": {"1", "2", "3"}
        }
        ref_pack_expectation: Dict[str, Set[str]] = {
            "Sentence": {"1", "2", "3"}
        }

        self.expected_types_and_attributes(pred_pack_expectation,
                                           ref_pack_expectation)
        self.check_record(pred_pack, ref_pack)
        self.writes_record(pred_pack, ref_pack)

    def get_result(self):
        pass


class DummyEvaluatorTwo(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def pred_pack_record(self, record_meta: Dict[str, Set[str]]):
        record_meta["Token"] = {"1", "2"}

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_pack_expectation: Dict[str, Set[str]] = {
            "Sentence": {"1", "2", "3"}
        }
        ref_pack_expectation: Dict[str, Set[str]] = {
            "Document": {"1", "2", "3"}
        }

        self.expected_types_and_attributes(pred_pack_expectation,
                                           ref_pack_expectation)
        self.check_record(pred_pack, ref_pack)
        self.writes_record(pred_pack, ref_pack)

    def get_result(self):
        pass


class DummyEvaluatorThree(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_pack_expectation: Dict[str, Set[str]] = {
            "ft.onto.example_import_ontology.Token": {"pos", "lemma"}
        }
        ref_pack_expectation: Dict[str, Set[str]] = {
            "ft.onto.example_import_ontology.Token": {"pos", "lemma"}
        }

        self.expected_types_and_attributes(pred_pack_expectation,
                                           ref_pack_expectation)
        self.check_record(pred_pack, ref_pack)
        self.writes_record(pred_pack, ref_pack)

    def get_result(self):
        pass


class RecordCheckPipelineTest(unittest.TestCase):

    def test_pipeline_reader_record_writing(self):
        """Tests reader record writing """

        nlp = Pipeline[DataPack](enforce_consistency=True)
        reader = DummySentenceReaderOne()
        nlp.set_reader(reader)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], {"1", "2", "3"})

    def test_pipeline_processor_record_writing(self):
        """Tests the processor record writing"""

        nlp = Pipeline[DataPack](enforce_consistency=True)
        reader = DummySentenceReaderOne()
        nlp.set_reader(reader)
        dummy = DummyPackProcessorOne()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], {"1", "2", "3"})
        self.assertEqual(pack._meta.record["Token"], {"1", "2"})
        self.assertEqual(pack._meta.record["Document"], {"2"})

    def test_pipeline_processor_record_checking_mismatching_error(self):
        """Tests the behavior of processor raising error exception
        and behavior of set enforce_consistency for the pipeline"""

        nlp = Pipeline[DataPack](enforce_consistency=True)
        reader = DummySentenceReaderOne()
        nlp.set_reader(reader)
        dummy = DummyPackProcessorTwo()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        with self.assertRaises(ProcessExecutionException):
            nlp.process(data_path)
        nlp.enforce_consistency(enforce=False)
        nlp.initialize()
        nlp.process(data_path)

    def test_pipeline_evaluator_record_writing(self):
        """Tests the evaluator record writing"""

        nlp = Pipeline[DataPack](enforce_consistency=True)
        reader = DummySentenceReaderOne()
        nlp.set_reader(reader)
        dummy = DummyEvaluatorOne()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], {"1", "2", "3"})
        self.assertEqual(pack._meta.record["Token"], {"1", "2"})

    def test_pipeline_evaluator_record_checking_mismatching_error(self):
        """Tests the behavior of evaluator raising error exception"""

        nlp = Pipeline[DataPack](enforce_consistency=True)
        reader = DummySentenceReaderOne()
        nlp.set_reader(reader)
        dummy = DummyEvaluatorTwo()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        with self.assertRaises(ProcessExecutionException):
            nlp.process(data_path)

    def test_reuse_processor(self):
        # Create a basic pipeline of multi packs that have two pack (by copying)
        nlp = Pipeline().set_reader(
            SentenceReader()).add(
            MultiPackBoxer()).add(
            MultiPackCopier())

        # Create one shared instance of this extractor
        dummy = DummyPackProcessor()
        nlp.add(dummy, config={"test": "dummy1"},
                selector=NameMatchSelector("default"))

        # This will not add the component successfully because the processor is
        # initialized.
        with self.assertRaises(ProcessorConfigError):
            nlp.add(dummy, config={"test": "dummy2"})

        # This will add the component, with a different selector
        nlp.add(dummy, selector=NameMatchSelector("copy"))
        nlp.initialize()

        # Check that the two processors have the same name.
        self.assertEqual(nlp.components[2].name,
                         get_full_module_name(DummyPackProcessor))
        self.assertEqual(nlp.components[3].name,
                         get_full_module_name(DummyPackProcessor))

        # Check that the two processors are also the same instance.
        self.assertEqual(nlp.components[2], nlp.components[3])

        # Check that the initialization is only done once, here the count
        #  will only be 1.
        self.assertEqual(nlp.components[2].initialize_count, 1)
        self.assertEqual(nlp.components[3].initialize_count, 1)

        # Check that the configuration is not changed by the second insertion.
        self.assertEqual(nlp.components[3].configs.test, 'dummy1')

        # Run it once to make sure it can run.
        dataset_path = os.path.join(data_samples_root, "random_texts", "0.txt")
        nlp.run(dataset_path)

        # Check that initialization will be false after `run`, because it
        #  calls the `finish` function of all components.
        self.assertFalse(nlp.components[2].is_initialized)
        self.assertFalse(nlp.components[3].is_initialized)

        # Check that we are able to re-initialize the pipeline.
        nlp.initialize()  # initialize the first time.
        nlp.initialize()  # re-initialize.

        # Check the name again after re-initialize.
        self.assertEqual(nlp.components[2].name,
                         get_full_module_name(DummyPackProcessor))
        self.assertEqual(nlp.components[3].name,
                         get_full_module_name(DummyPackProcessor))

        # Obtain the results from the multipack.
        mp: MultiPack = nlp.process(dataset_path)
        pack: DataPack = mp.get_pack("default")
        pack_copy: DataPack = mp.get_pack("copy")

        # Check both pack are processed by the DummyProcessor once, because
        #  we use different selector.
        pack.get_single(NewType).value = "[PACK]"
        pack_copy.get_single(NewType).value = "[PACK]"

    def test_pipeline_processor_subclass_type_checking(self):
        r"""Tests the processor record subclass type checking for processor with
        pipeline initialized with ontology specification file"""
        onto_specs_file_path = os.path.join(onto_specs_samples_root,
                                            'example_merged_ontology.json')
        nlp = Pipeline[DataPack](ontology_file=onto_specs_file_path,
                                 enforce_consistency=True)
        reader = DummySentenceReaderTwo()
        nlp.set_reader(reader)
        dummy = DummyPackProcessorThree()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record, {
            "ft.onto.example_ontology.Word": {"string_features",
                                              "word_forms",
                                              "token_ranks"}
        })

    def test_pipeline_evaluator_subclass_type_checking(self):
        r"""Tests the processor record subclass type checking for evaluator with
        pipeline initialized with ontology specification file"""
        onto_specs_file_path = os.path.join(onto_specs_samples_root,
                                            'example_merged_ontology.json')
        nlp = Pipeline[DataPack](ontology_file=onto_specs_file_path,
                                 enforce_consistency=True)
        reader = DummySentenceReaderTwo()
        nlp.set_reader(reader)
        dummy = DummyEvaluatorThree()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record, {
            "ft.onto.example_ontology.Word": {"string_features",
                                              "word_forms",
                                              "token_ranks"}
        })


if __name__ == '__main__':
    unittest.main()
