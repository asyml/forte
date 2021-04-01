# Copyright 2021 The Forte Authors. All Rights Reserved.
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
Unit tests for Pipeline with Record Check functions-.
"""

import os
import unittest
from typing import Dict, List

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from forte.common import ProcessExecutionException
from forte.pipeline import Pipeline

from tests.forte.pipeline_test import SentenceReader, DummyPackProcessor


data_samples_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    *([os.path.pardir] * 2), 'data_samples'))


class DummySentenceReader(SentenceReader):

    def record(self, record_meta: Dict[str, List[str]]):
        record_meta["Sentence"] = ["1", "2", "3"]


class DummyPackProcessorOne(DummyPackProcessor):

    def record(self, record_meta: Dict[str, List[str]]):
        record_meta["Token"] = ["1", "2"]
        record_meta["Document"] = ["2"]

    @classmethod
    def expected_types_and_attributes(cls):
        expectation = dict()
        expectation["Sentence"] = ["1", "2", "3"]
        return expectation


class DummyPackProcessorTwo(DummyPackProcessor):

    def record(self, record_meta: Dict[str, List[str]]):
        record_meta["Token"] = ["1", "2"]
        record_meta["Document"] = ["2"]

    @classmethod
    def expected_types_and_attributes(cls):
        expectation = dict()
        expectation["Document"] = ["1", "2", "3", "4"]
        return expectation


class DummyEvaluatorOne(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def pred_pack_record(self, record_meta: Dict[str, List[str]]):
        record_meta["Token"] = ["1", "2"]

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_pack_expectation = dict()
        pred_pack_expectation["Sentence"] = ["1", "2", "3"]
        ref_pack_expectation = dict()
        ref_pack_expectation["Sentence"] = ["1", "2", "3"]
        self.expected_types_and_attributes(pred_pack_expectation,
                                           ref_pack_expectation)
        self.check_record(pred_pack, ref_pack)
        self.writes_record(pred_pack, ref_pack)

    def get_result(self):
        pass


class DummyEvaluatorTwo(Evaluator):
    """ This evaluator does nothing, just for test purpose."""

    def pred_pack_record(self, record_meta: Dict[str, List[str]]):
        record_meta["Token"] = ["1", "2"]

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_pack_expectation = dict()
        pred_pack_expectation["Sentence"] = ["1", "2", "3"]
        ref_pack_expectation = dict()
        ref_pack_expectation["Document"] = ["1", "2", "3"]
        self.expected_types_and_attributes(pred_pack_expectation,
                                           ref_pack_expectation)
        self.check_record(pred_pack, ref_pack)
        self.writes_record(pred_pack, ref_pack)

    def get_result(self):
        pass


class RecordCheckPipelineTest(unittest.TestCase):

    def test_pipeline1(self):
        """Tests reader record writing """

        nlp = Pipeline[DataPack]()
        nlp.enforce_consistency(enforce=True)
        reader = DummySentenceReader()
        nlp.set_reader(reader)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], ["1", "2", "3"])

    def test_pipeline2(self):
        """Tests the processor record writing"""

        nlp = Pipeline[DataPack]()
        nlp.enforce_consistency(enforce=True)
        reader = DummySentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessorOne()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], ["1", "2", "3"])
        self.assertEqual(pack._meta.record["Token"], ["1", "2"])
        self.assertEqual(pack._meta.record["Document"], ["2"])

    def test_pipeline3(self):
        """Tests the behavior of processor raising error exception"""

        nlp = Pipeline[DataPack]()
        nlp.enforce_consistency(enforce=True)
        reader = DummySentenceReader()
        nlp.set_reader(reader)
        dummy = DummyPackProcessorTwo()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        with self.assertRaises(ProcessExecutionException):
            nlp.process(data_path)

    def test_pipeline4(self):
        """Tests the evaluator record writing"""

        nlp = Pipeline[DataPack]()
        nlp.enforce_consistency(enforce=True)
        reader = DummySentenceReader()
        nlp.set_reader(reader)
        dummy = DummyEvaluatorOne()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        pack = nlp.process(data_path)
        self.assertEqual(pack._meta.record["Sentence"], ["1", "2", "3"])
        self.assertEqual(pack._meta.record["Token"], ["1", "2"])

    def test_pipeline5(self):
        """Tests the behavior of evaluator raising error exception"""

        nlp = Pipeline[DataPack]()
        nlp.enforce_consistency(enforce=True)
        reader = DummySentenceReader()
        nlp.set_reader(reader)
        dummy = DummyEvaluatorTwo()
        nlp.add(dummy)
        nlp.initialize()
        data_path = data_samples_root + "/random_texts/0.txt"
        self.assertRaises(ProcessExecutionException,
                          nlp.process, data_path)


if __name__ == '__main__':
    unittest.main()
