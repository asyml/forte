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
import errno

from typing import Iterator, Dict, List

from ft.onto.base_ontology import (
    Sentence, ConstituentNode)

from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader

__all__ = [
    "SST2Reader"
]


class SST2Reader(PackReader):
    r""":class:`SST2Reader` is designed to read in the
    Stanford Sentiment Treebank 2 dataset from
    https://nlp.stanford.edu/sentiment/treebank.html
    To use the dataset, please download it from the webpage.

    Provided the ss2_dir_path, the _collect function will look for the following files:
        "dictionary.txt": a mapping from phrase text to phrase id
        "sentiment_labels": a mapping from phrase id to sentiment score
        "datasetSentences.txt": original text for the sentence
        "STree.txt": parent pointer list of constituency tree for each sentence

    """
    def __init__(self):
        super().__init__()
        # Transform the text-form phrase to sentiment score.
        self.phrase_to_id: Dict = {}
        self.id_to_senti: Dict = {}

    def _cache_key_function(self, data_pack: DataPack) -> str:
        if data_pack.meta.pack_name is None:
            raise ValueError("data_pack does not have a sentence id")
        return data_pack.meta.pack_name

    def _check_file_exist(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename)

    def _collect(self, *args, **kwargs) -> Iterator[DataPack]:
        # pylint: disable = unused-argument
        r"""Iterator over sst files in the data_source.
        The directory should at least have the following files:
            "dictionary.txt": a mapping from phrase text to phrase id
            "sentiment_labels": a mapping from phrase id to sentiment score
            "datasetSentences.txt": original text for the sentence
            "STree.txt": parent pointer list of constituency tree for each sentence
        Args:
            args: args[0] is the directory to the sst2 files.
            kwargs:
        Returns: data packs obtained from each sentence from the sst2 file.
        """
        sent_id: str
        sent_text: str
        parent_pointer_list: List[int]

        sst2_dir_path: str = args[0]
        phrase_to_id_path: str = os.path.join(sst2_dir_path, "dictionary.txt")
        id_to_senti_path: str = os.path.join(sst2_dir_path, "sentiment_labels.txt")
        text_path: str = os.path.join(sst2_dir_path, "datasetSentences.txt")
        tree_path: str = os.path.join(sst2_dir_path, "STree.txt")
        self._check_file_exist(phrase_to_id_path)
        self._check_file_exist(id_to_senti_path)
        self._check_file_exist(text_path)
        self._check_file_exist(tree_path)

        # Read the mapping from phrase to phrase-id.
        with open(phrase_to_id_path, "r", encoding="utf8") as file:
            for line in file:
                phrase, id = line.split("|")
                self.phrase_to_id[phrase] = int(id)

        # Read the mapping from phrase-id to sentiment score.
        with open(id_to_senti_path, "r", encoding="utf8") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                id, score = line.split("|")
                self.id_to_senti[int(id)] = float(score)

        # Read the text and tree structure.
        with open(text_path, "r", encoding="utf8") as ftext, open(tree_path, "r", encoding="utf8") as ftree:
            ftext.readline()  # Skip the headers.
            for line_text, line_tree in zip(ftext, ftree):
                sent_id, sent_text = line_text.split("\t")
                parent_pointer_list = line_tree.split("|")
                parent_pointer_list = list(map(int, parent_pointer_list))
                yield (sent_id, sent_text, parent_pointer_list)

    def _get_span_with_dfs(self, span_begin_end: List[List[int]], children_nodes: List[List[int]], cur_node: int):
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


    def _parse_parent_pointer_list(self, data_pack: DataPack, sent_text: str, parent_pointer_list: List[int]):
        r"""Build the ConstituentNode objects from parent pointer list.
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

        # Get the children node ids for each node, node 0 is the root.
        children_nodes: List[List[int]] = [[] for _ in range(n_nodes)]
        for i in range(1, n_nodes):
            parent = parent_pointer_list[i-1]
            children_nodes[parent].append(i)

        # Get the begin/end index of spans for leaf nodes.
        for i in range(1, len(tokens)+1):
            span_begin = 0
            if i > 1:
                span_begin = span_begin_end[i-1][1] + 1  # plus 1 for the whitespace separator
            span_end = span_begin + len(tokens[i-1])
            span_begin_end[i] = [span_begin, span_end]

        # Get the spans recursively and store in "span_begin_end".
        self._get_span_with_dfs(span_begin_end, children_nodes, 0)

        # Create the constituency Tree.
        node_list = [ConstituentNode(data_pack, begin, end) for (begin, end) in span_begin_end]
        # Get the sentiment scores.
        for i in range(n_nodes):
            phrase = node_list[i].text
            phrase_id = self.phrase_to_id.get(phrase, -1)
            node_list[i].sentiment["pos"] = self.id_to_senti.get(phrase_id, 0.5)

        # Link the parent and children nodes.
        for i in range(1, n_nodes):
            parent = parent_pointer_list[i-1]
            node_list[i].parent_node = node_list[parent]

        for i in range(n_nodes):
            # Sort the children nodes by span begin position.
            children_nodes[i].sort(key=lambda x: node_list[x].begin)
            for child in children_nodes[i]:
                node_list[i].children_nodes.append(node_list[child])

        # Set the is_leaf/is_root flag.
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

        # Add sentence to data_pack.
        Sentence(data_pack, 0, len(sent_text))

        self._parse_parent_pointer_list(data_pack, sent_text, parent_pointer_list)
        yield data_pack
