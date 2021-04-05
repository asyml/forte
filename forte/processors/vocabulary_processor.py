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

import json
import os
from abc import ABC
from typing import Counter as CounterType, Dict, List, Optional

import texar.torch as tx

from forte.processors.base import PackProcessor

__all__ = [
    "Alphabet",
    "VocabularyProcessor",
]


class Alphabet:
    """
    Args:
        name: The name of the alphabet
        word_cnt: Initial vocabulary counting
        keep_growing: If True, new instances not found ruing `get_index` will
            be added to the vocabulary.
        ignore_case_in_query:
            If it's True, Alphabet will try to query the lower-cased input from
            it's vocabulary if it cannot find the input in its keys.
    """

    def __init__(
            self,
            name,
            word_cnt: Optional[CounterType[str]] = None,
            keep_growing: bool = True,
            ignore_case_in_query: bool = True,
    ):
        self.__name = name
        self.reserved_tokens = tx.data.SpecialTokens

        self.instance2index: Dict = {}
        self.instances: List = []

        for sp in [
            self.reserved_tokens.PAD,
            self.reserved_tokens.BOS,
            self.reserved_tokens.EOS,
            self.reserved_tokens.UNK,
        ]:
            self.instance2index[sp] = len(self.instance2index)
            self.instances.append(sp)

        self.pad_id = self.instance2index[self.reserved_tokens.PAD]  # 0
        self.bos_id = self.instance2index[self.reserved_tokens.BOS]  # 1
        self.eos_id = self.instance2index[self.reserved_tokens.EOS]  # 2
        self.unk_id = self.instance2index[self.reserved_tokens.UNK]  # 3

        self.keep_growing = keep_growing
        self.ignore_case_in_query = ignore_case_in_query

        if word_cnt is not None:
            for word in word_cnt:
                self.add(word)
            self.close()

    def add(self, instance):
        if instance not in self.instance2index:
            self.instance2index[instance] = len(self.instance2index)
            self.instances.append(instance)

    def get_index(self, instance):
        """
        Args:
            instance: the input token

        Returns:
            the index of the queried token in the dictionary
        """
        if instance is None:
            return self.instance2index[self.reserved_tokens.PAD]

        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                self.add(instance)
                return self.instance2index[instance]
            else:
                if self.ignore_case_in_query:
                    try:
                        return self.instance2index[instance.lower()]
                    except KeyError:
                        return self.instance2index[self.reserved_tokens.UNK]
                else:
                    return self.instance2index[self.reserved_tokens.UNK]

    def get_instance(self, index):
        try:
            return self.instances[index]
        except IndexError as e:
            raise IndexError("unknown index: %d" % index) from e

    def size(self):
        return len(self.instances)

    def items(self):
        return self.instance2index.items()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {
            "instance2index": self.instance2index,
            "instances": self.instances,
        }

    def __from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alphabet records to the given directory.

        Args:
            output_directory: Directory to save model and weights.
            name: The alphabet saving name, optional.
        """
        saving_name = name if name else self.__name

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        json.dump(
            self.get_content(),
            open(os.path.join(output_directory, saving_name + ".json"), "w"),
            indent=4,
        )

    def load(self, input_directory, name=None):
        loading_name = name if name else self.__name
        self.__from_json(
            json.load(
                open(os.path.join(input_directory, loading_name + ".json"))
            )
        )
        self.keep_growing = False


class VocabularyProcessor(PackProcessor, ABC):
    """
    Build vocabulary from the input DataPack, write the result into the
    shared resources.
    """

    def __init__(self) -> None:
        super().__init__()
        self.min_frequency = 0
