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
Unit tests for remote processor.
"""

import os
import unittest
from typing import Dict, Set, Any, Iterator

from ddt import ddt, data

from forte.data.base_reader import MultiPackReader
from forte.data.common_entry_utils import create_utterance, get_last_utterance
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.code_generation_objects import EntryTreeNode
from forte.data.ontology.top import Generics
from forte.data.readers import StringReader
from forte.data.selector import RegexNameMatchSelector
from forte.pipeline import Pipeline
from forte.processors.base import PackProcessor
from forte.processors.nlp import ElizaProcessor
from ft.onto.base_ontology import Utterance

TEST_RECORDS_1 = {
    "Token": {"1", "2"},
    "Document": {"2"},
}
TEST_RECORDS_2 = {
    "ft.onto.example_import_ontology.Token": {"pos", "lemma"},
    "Sentence": {"1", "2", "3"},
}


class UserSimulator(PackProcessor):
    """
    A simulated processor that will generate utterance based on the config.
    """

    def _process(self, input_pack: DataPack):
        create_utterance(input_pack, self.configs.user_input, "user")

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config["user_input"] = ""
        return config


class DummyMultiPackReader(MultiPackReader):
    def _collect(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        yield 0

    def _parse_pack(self, collection: Any) -> Iterator[MultiPack]:
        multi_pack: MultiPack = MultiPack()
        data_pack1 = multi_pack.add_pack(ref_name="pack1")
        data_pack2 = multi_pack.add_pack(ref_name="pack2")
        data_pack3 = multi_pack.add_pack(ref_name="pack_three")

        data_pack1.pack_name = "1"
        data_pack2.pack_name = "2"
        data_pack3.pack_name = "Three"
        yield multi_pack


class DummyProcessor(PackProcessor):
    """
    A dummy Processor to check the expected/output records from the remote
    pipeline.
    """

    def __init__(
        self,
        expected_records: Dict[str, Set[str]] = {},
        output_records: Dict[str, Set[str]] = {},
    ):
        self._expected_records: Dict[str, Set[str]] = expected_records
        self._output_records: Dict[str, Set[str]] = output_records

    def _process(self, input_pack: DataPack):
        entries = list(input_pack.get_entries_of(Generics))
        if len(entries) == 0:
            Generics(pack=input_pack)
        else:
            entry = entries[0]

    def expected_types_and_attributes(self):
        return self._expected_records

    def record(self, record_meta: Dict[str, Set[str]]):
        record_meta.update(self._output_records)


@ddt
class AdvancedPipelineTest(unittest.TestCase):
    """
    Test intermediate representation. Here we use eliza
    pipeline as an example, and all the testcases below are refactored from
    `eliza_test.py`.
    """

    def setUp(self) -> None:
        dir_path: str = os.path.dirname(os.path.abspath(__file__))
        self._pl_config_path: str = os.path.join(dir_path, "eliza_pl_ir.yaml")
        self._onto_path: str = os.path.join(
            dir_path, "data/ontology/test_specs/base_ontology.json"
        )

    @data(
        [
            "I would like to have a chat bot.",
            "You say you would like to have a chat bot ?",
        ],
        ["bye", "Goodbye.  Thank you for talking to me."],
    )
    def test_ir_basic(self, input_output_pair):
        """
        Verify the intermediate representation of pipeline.
        """
        i_str, o_str = input_output_pair

        # Build eliza pipeline
        eliza_pl: Pipeline[DataPack] = Pipeline[DataPack](
            ontology_file=self._onto_path,
            enforce_consistency=True,
            do_init_type_check=True,
        )
        eliza_pl.set_reader(StringReader())
        eliza_pl.add(UserSimulator(), config={"user_input": i_str})
        eliza_pl.add(ElizaProcessor())
        eliza_pl.set_profiling()
        eliza_pl.initialize()
        eliza_pl.save(self._pl_config_path)

        # Build test pipeline
        test_pl: Pipeline[DataPack] = Pipeline[DataPack]()
        test_pl.init_from_config_path(self._pl_config_path)

        # Verify pipeline states
        self.assertListEqual(
            *map(
                lambda pl: [
                    getattr(pl, attr)
                    for attr in (
                        "_initialized",
                        "_enable_profiling",
                        "_check_type_consistency",
                        "_do_init_type_check",
                    )
                    if hasattr(pl, attr)
                ],
                (eliza_pl, test_pl),
            )
        )
        self.assertDictEqual(
            eliza_pl.resource.get("onto_specs_dict"),
            test_pl.resource.get("onto_specs_dict"),
        )
        self._assertEntryTreeEqual(
            eliza_pl.resource.get("merged_entry_tree").root,
            test_pl.resource.get("merged_entry_tree").root,
        )

        # Verify output
        test_pl.initialize()
        res: DataPack = test_pl.process("")
        utterance = get_last_utterance(res, "ai")
        self.assertEqual(len([_ for _ in res.get(Utterance)]), 2)
        self.assertEqual(utterance.text, o_str)

    def test_ir_selector(self):
        """
        Test the intermediate representation of selector.
        """
        # Build original pipeline with RegexNameMatchSelector
        pl: Pipeline = Pipeline[MultiPack]()
        pl.set_reader(DummyMultiPackReader())
        pl.add(
            DummyProcessor(),
            selector=RegexNameMatchSelector(),
            selector_config={"select_name": "^.*\\d$"},
        )
        pl.save(self._pl_config_path)

        # Verify the selector from IR
        test_pl: Pipeline = Pipeline[MultiPack]()
        test_pl.init_from_config_path(self._pl_config_path)
        test_pl.initialize()
        for multi_pack in test_pl.process_dataset():
            for _, pack in multi_pack.iter_packs():
                self.assertEqual(
                    pack.num_generics_entries, int(pack.pack_name in ("1", "2"))
                )

    def _assertEntryTreeEqual(self, root1: EntryTreeNode, root2: EntryTreeNode):
        """
        Test if two `EntryTreeNode` objects are recursively equivalent
        """
        self.assertEqual(root1.name, root2.name)
        self.assertSetEqual(root1.attributes, root2.attributes)
        self.assertEqual(len(root1.children), len(root2.children))
        for i in range(len(root1.children)):
            self._assertEntryTreeEqual(root1.children[i], root2.children[i])

    def tearDown(self) -> None:
        """
        Remove the IR file of pipeline if necessary.
        """
        if os.path.exists(self._pl_config_path):
            os.remove(self._pl_config_path)


if __name__ == "__main__":
    unittest.main()
