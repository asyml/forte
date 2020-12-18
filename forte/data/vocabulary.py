# Copyright 2020 The Forte Authors. All Rights Reserved.
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
# pylint: disable=line-too-long
from typing import List, Tuple, Dict, Union, Hashable, Iterable


class Vocabulary:
    """This class will store "Elements" that are added, assign "Ids" to them and
    return "Representations" if queried. These three are the main concepts in this class.

    1. Element: Any hashable instance that the user want to store.
    2. Id: Each element will have an unique Id, which is an interger.
    3. Representation: according to the configuration, the representation for an element
        could be an interger (in this case, would be "Id"), or an one-hot vector (in this
        case, would be a list of interger).

    There are two special elements.

    1. One is <PAD> element, which will be mapped into Id of 0 or -1 and have different
        representation according to different setting.
    2. The other one is <UNK> element, which, if added into the vocabulary, will be the
        default element if the queried element is not found.

    Here is a table on how our Vocabulary class behavior under different settings.

    User_request = {
        "vocab method":  "raw"(handle outside), "indexing",      "indexing",          "one-hot",                    "one-hot"
        "need_pad":           assume False,        True,            False,              True,                         False
    }

    Vocabulary behavior:
        get_pad_value            None                0              None                [0,0,0]                        None
        inner_mapping            None           0:pad  1:ele0      0:ele0         -1:<PAD>   0:ele0                   0:ele0
        element2repr           raise Error      pad->0 ele0->1     ele0->0      <PAD>->[0,0,0] ele0->[1,0,0]        ele0->[1,0,0]
        id2element             raise Error      0->pad 1->ele0     0->ele0    -1 -> <PAD> 0->ele0 (be careful)        0->ele0

    Not implemented for simplicity:
        element2id             raise Error      pad->0 ele0->1     ele0->0        <PAD> -> -1   ele0 -> 0             ele0->0
        repr2element           raise Error,     0->pad 1->ele0     0->ele0     [0,0,0]-><PAD> [1,0,0]->ele0        [1,0,0]->ele0
    """
    PAD_ELEMENT = "<PAD>"
    UNK_ELEMENT = "<UNK>"

    def __init__(self, method: str, need_pad: bool, use_unk: bool):
        self.method = method
        self.need_pad = need_pad
        self.use_unk = use_unk

        self.element2id_dict: Dict = dict()
        self.id2element_dict: Dict = dict()

        if method == "one-hot" and need_pad:
            self.next_id = -1
        else:
            self.next_id = 0

        if need_pad:
            self.add_element(Vocabulary.PAD_ELEMENT)

        if use_unk:
            self.add_element(Vocabulary.UNK_ELEMENT)

    def add_element(self, element: Hashable):
        if element not in self.element2id_dict:
            self.element2id_dict[element] = self.next_id
            self.id2element_dict[self.next_id] = element
            self.next_id += 1

    def id2element(self, idx: int) -> Hashable:
        return self.id2element_dict[idx]

    def element2repr(self, element: Hashable) \
                    -> Union[int, List[int]]:
        if self.use_unk:
            idx = self.element2id_dict.get(element,
                    self.element2id_dict[Vocabulary.UNK_ELEMENT])
        else:
            idx = self.element2id_dict[element]

        if self.method == "indexing":
            return idx
        else:
            vec_size = len(self.element2id_dict)
            if self.need_pad:
                vec_size -= 1
            vec = [0 for _ in range(vec_size)]
            if idx != -1:
                vec[idx] = 1
            return vec

    def __len__(self) -> int:
        return len(self.element2id_dict)

    def has_element(self, element: Hashable) -> bool:
        return element in self.element2id_dict

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        return self.element2id_dict.items()

    def get_dict(self) -> Dict[Hashable, int]:
        return self.element2id_dict

    def get_pad_value(self) -> Union[None, int, List[int]]:
        if self.need_pad:
            return self.element2repr(self.PAD_ELEMENT)
        return None
