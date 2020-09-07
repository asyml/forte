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
The reader that reads Stanford Sentiment Treebank
https://nlp.stanford.edu/sentiment/treebank.html
into data_pack format
"""
import os

from typing import Iterator, Dict, Tuple, Any

from ft.onto.base_ontology import (
    Sentence, Token, ConstituentNode)

from forte.data.data_utils_io import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader

__all__ = [
    "SST2Reader"
]


class SST2Reader(PackReader):
    r""":class:`SST2Reader` is designed to read in the 
    Stanford Sentiment Treebank 2 dataset
    """
    def __init__(self):
        super().__init__()
        # transform the text-form phrase to sentiment score
        self.phrase_to_id: Dict = {}
        self.id_to_senti: Dict = {}

    def _cache_key_function(self, data_pack: Any) -> str:
        if data_pack.meta.pack_name is None:
            raise ValueError("data_pack does not have a sentence id")
        return data_pack.meta.pack_name

    def _collect(self, *args, **kwargs) -> Iterator[Any]:
        # pylint: disable = unused-argument
        r"""Iterator over sst files in the data_source.
        Args:
            args: args[0] is the directory to the sst2 files.
            kwargs:
        Returns: data packs obtained from each sentence from the sst2 file.
        """
        sent_id: str
        sent_text: str
        parent_pointer_list: List[int]

        sst2_dir_path = args[0]
        phrase_to_id_path = os.path.join(sst2_dir_path, "dictionary.txt")
        id_to_senti_path = os.path.join(sst2_dir_path, "sentiment_labels.txt")
        text_path = os.path.join(sst2_dir_path, "datasetSentences.txt")
        tree_path = os.path.join(sst2_dir_path, "STree.txt")

        # read the mapping from phrase to phrase-id
        with open(phrase_to_id_path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                phrase, id = line.split("|")
                self.phrase_to_id[phrase] = int(id)

        # read the mapping from phrase-id to sentiment score
        with open(id_to_senti_path, "r", encoding="utf8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                id, score = line.split("|")
                self.id_to_senti[int(id)] = float(score)

        # read the text and tree structure
        with open(text_path, "r", encoding="utf8") as ftext, open(tree_path, "r", encoding="utf8") as ftree:
            texts = ftext.readlines()
            for i, line in enumerate(texts):
                if i == 0:  # skip the headers
                    continue
                sent_id, sent_text = line.split("\t")
                parent_pointer_list = ftree.readline().split("|")
                parent_pointer_list = list(map(int, parent_pointer_list))
                yield (sent_id, sent_text, parent_pointer_list)

    def _get_span_with_dfs(self, span_begin_end, children_nodes, cur_node):
        r"""Recursively get the span for each node in the tree
        Args:
            span_begin_end: stores the span (begin, end) posititon
            children_nodes: the structure of the tree
            cur_node: current processing node
        Returns: None
        """
        if len(children_nodes[cur_node]) == 0:
            return
        begin = -1
        end = -1
        for child in children_nodes[cur_node]:
            self._get_span_with_dfs(span_begin_end, children_nodes, child)
            if begin == -1 or begin > span_begin_end[child][0]:
                begin = span_begin_end[child][0]
            if end == -1 or end < span_begin_end[child][1]:
                end = span_begin_end[child][1]
        span_begin_end[cur_node] = [begin, end]


    def _parse_parent_pointer_list(self, data_pack, sent_text, parent_pointer_list):
        r"""build the ConstituentNode objects from parent pointer list.
        Args:
            data_pack: the data_pack to add ConstituentNode
            sent_text: the whitespace-splitted sentence text
            parent_pointer_list: the parent pointer list is a format to store the constituency tree
        Returns: None
        """

        tokens: List[str] = sent_text.split()
        n_nodes: int = len(parent_pointer_list) + 1
        span_begin_end: List[List[int]] = [None] * n_nodes
        node_list: List[ConstituentNode] = [None] * n_nodes

        # get the children node ids for each node, node 0 is the root
        children_nodes = [[] for _ in range(n_nodes)]
        for i in range(1, n_nodes):
            parent = parent_pointer_list[i-1]
            children_nodes[parent].append(i)

        # get the begin/end index of spans for leaf nodes
        for i in range(1, len(tokens)+1):
            span_begin = 0
            if i > 1:
                span_begin = span_begin_end[i-1][1] + 1  # plus 1 for the whitespace separator
            span_end = span_begin + len(tokens[i-1])
            span_begin_end[i] = [span_begin, span_end]

        # get the spans recursively and store in "span_begin_end"
        self._get_span_with_dfs(span_begin_end, children_nodes, 0)

        # create the constituency Tree
        node_list = [ConstituentNode(data_pack, begin, end) for (begin, end) in span_begin_end]
        # get the sentiment scores
        for i in range(n_nodes):
            phrase = node_list[i].text
            phrase_id = self.phrase_to_id.get(phrase, -1)
            node_list[i].sentiment["pos"] = self.id_to_senti.get(phrase_id, 0.5)

        # link the parent and children nodes
        for i in range(1, n_nodes):
            parent = parent_pointer_list[i-1]
            node_list[i].parent_node = node_list[parent]

        for i in range(n_nodes):
            # sort the children nodes by span begin position
            children_nodes[i].sort(key=lambda x: node_list[x].begin)
            for child in children_nodes[i]:
                node_list[i].children_nodes.append(node_list[child])

        # Set the is_leaf/is_root flag
        for i in range(n_nodes):
            node_list[i].is_leaf = False
            node_list[i].is_root = False

        for i in range(1, len(tokens)+1):
            node_list[i].is_leaf = True
        node_list[0].is_root = True

    def _parse_pack(self, sent_line) -> Iterator[DataPack]:
        data_pack: DataPack = self.new_pack()

        sent_id: str = sent_line[0]
        sent_text: str = sent_line[1].strip()
        parent_pointer_list: List[int] = sent_line[2]

        data_pack.pack_name = sent_id
        data_pack.set_text(sent_text)

        # add sentence to data_pack
        Sentence(data_pack, 0, len(sent_text))

        self._parse_parent_pointer_list(data_pack, sent_text, parent_pointer_list)
        yield data_pack