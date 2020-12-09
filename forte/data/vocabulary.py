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


from typing import List, Tuple, Dict, Union, Hashable, Iterable


class Vocabulary:
    '''This class maps element to representation. Element
    could be any hashable type and there are two types of
    representations, namely, "indexing" and "one-hot". There
    are two special types of element, namely PAD_ELEMENT
    and UNK_ELEMENT.
    For "indexing" vocabulary,
        Element:  <PAD>  ele1   ele2   ele3  ...
        Id:         0      1      2      3   ...
        Repr:       0      1      2      3   ...
    For "one-hot" vocabulary,
        Element:  <PAD>  ele1   ele2   ele3  ...
        Id:        -1      0      1      2   ...
        Repr:      [0,    [1,    [0,    [0,  ...
                    0,     0,     1,     0,  ...
                    0,     0,     0,     1,  ...
                    0,     0,     0,     0,  ...
                    ...]   ...]   ...]   ...]
    If vocabulary uses UNK_ELEMENT, the first element,
    "ele1" will be UNK_ELEMENT and any other elements
    that cannot be found in the current vocabulary will
    be mapped to the UNK_ELEMENT. Otherwise, UNK_ELEMENT
    is not used. Error will occur when querying unknown
    element in the vocabulary.
    '''
    PAD_ELEMENT = "<PAD>"
    UNK_ELEMENT = "<UNK>"

    def __init__(self, method: str, use_unk: bool):
        self.element2id_dict = dict()
        self.id2element_dict = dict()

        if method == "indexing":
            self.next_id = 0
        elif method == "one-hot":
            self.next_id = -1
        else:
            raise AttributeError("The method %s \
                is not supported in Vocabulary!" % method)

        self.add(Vocabulary.PAD_ELEMENT)

        if use_unk:
            self.add(Vocabulary.UNK_ELEMENT)

        self.method = method
        self.use_unk = use_unk

    def add(self, element: Hashable):
        if element not in self.element2id_dict:
            self.element2id_dict[element] = self.next_id
            self.id2element_dict[self.next_id] = element
            self.next_id += 1

    def id2repr(self, idx: int) -> List[int]:
        if self.method == "indexing":
            return idx
        vec = [0] * self.next_id
        if idx == -1:
            return vec
        else:
            vec[idx] = 1
            return vec

    def element2repr(self, element: Hashable) \
                    -> Union[int, List[int]]:
        if self.use_unk:
            idx = self.element2id_dict.get(element,
                    self.element2id_dict[Vocabulary.UNK_ELEMENT])
        else:
            idx = self.element2id_dict[element]
        return self.id2repr(idx)

    def id2element(self, idx: int) -> Hashable:
        return self.id2element_dict[idx]

    def __len__(self) -> int:
        if self.method == "indexing":
            return len(self.element2id_dict)
        else:
            return len(self.element2id_dict) - 1

    def has_element(self, element: Hashable) -> bool:
        return element in self.element2id_dict

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        return self.element2id_dict.items()

    def get_dict(self) -> Dict[Hashable, int]:
        return self.element2id_dict
