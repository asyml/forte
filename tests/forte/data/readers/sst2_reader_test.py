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
Unit tests for SST2Reader.
"""
import json
import os
import unittest
from typing import Iterator, Iterable

from forte.data.data_pack import DataPack
from forte.data.readers import SST2Reader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import ConstituentNode

class SST2ReaderTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/sst2'))

    def test_reader(self):
        pipeline = Pipeline()
        reader = SST2Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        data_packs: Iterable[DataPack] = pipeline.process_dataset(
            self.dataset_path)
        file_paths: Iterator[str] = reader._collect(self.dataset_path)

        # test the count of all sentences
        count_pack: int = 0
        for i, pack in enumerate(data_packs):
            count_pack += 1
            # test a simple case
            """
                                0
                                |
                                7
                          ______|_______
                          |            |
                          6            5
                     _____|____   _____|_____
                     |        |   |         |
                Effective   but  too-tepid  biopic 
            """
            if i == 2:
                count_root: int = 0
                count_leaf: int = 0
                root: ConstituentNode = None
                for cnode in pack.get(ConstituentNode):
                    if cnode.is_root:
                        root = cnode
                        count_root += 1
                    if cnode.is_leaf:
                        count_leaf += 1
                self.assertEqual(count_root, 1)
                self.assertEqual(count_leaf, 4)

                # node 0
                self.assertEqual(root.text, "Effective but too-tepid biopic")
                self.assertEqual(len(root.children_nodes), 1)

                # node 7
                root = root.children_nodes[0]
                self.assertEqual(root.text, "Effective but too-tepid biopic")
                self.assertEqual(len(root.children_nodes), 2)
                self.assertEqual(root.sentiment["pos"], 0.51389)

                left_subtree = root.children_nodes[0]
                right_subtree = root.children_nodes[1]

                # node 6
                self.assertEqual(left_subtree.text, "Effective but")
                self.assertEqual(len(left_subtree.children_nodes), 2)
                self.assertEqual(left_subtree.sentiment["pos"], 0.63889)
                # node 5
                self.assertEqual(right_subtree.text, "too-tepid biopic")
                self.assertEqual(len(right_subtree.children_nodes), 2)
                self.assertEqual(right_subtree.sentiment["pos"], 0.375)

                leaf_node_1 = left_subtree.children_nodes[0]
                leaf_node_2 = left_subtree.children_nodes[1]
                leaf_node_3 = right_subtree.children_nodes[0]
                leaf_node_4 = right_subtree.children_nodes[1]

                self.assertEqual(leaf_node_1.text, "Effective")
                self.assertEqual(leaf_node_1.is_leaf, True)
                self.assertEqual(leaf_node_1.parent_node, left_subtree)

                self.assertEqual(leaf_node_2.text, "but")
                self.assertEqual(leaf_node_2.is_leaf, True)
                self.assertEqual(leaf_node_2.parent_node, left_subtree)

                self.assertEqual(leaf_node_3.text, "too-tepid")
                self.assertEqual(leaf_node_3.is_leaf, True)
                self.assertEqual(leaf_node_3.parent_node, right_subtree)

                self.assertEqual(leaf_node_4.text, "biopic")
                self.assertEqual(leaf_node_4.is_leaf, True)
                self.assertEqual(leaf_node_4.parent_node, right_subtree)
            print(i)
        self.assertEqual(count_pack, 11855)


if __name__ == "__main__":
    unittest.main()
