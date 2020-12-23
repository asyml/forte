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
    r"""This class will store "Elements" that are added, assign "Ids" to them and
    return "Representations" if queried. These three are the main concepts in this class.

    1. Element: Any hashable instance that the user want to store.
    2. Id: Each element will have an unique Id, which is an integer.
    3. Representation: according to the configuration, the representation for an element
        could be an integer (in this case, would be "Id"), or an one-hot vector (in this
        case, would be a list of integer).

    There are two special elements.

    1. One is <PAD> element, which will be mapped into Id of 0 or -1 and have different
        representation according to different setting.
    2. The other one is <UNK> element, which, if added into the vocabulary, will be the
        default element if the queried element is not found.

    Here is a table on how our Vocabulary class behavior under different settings. Element0 means
    the first element that is added to the vocabulary. Elements added later will be element1,
    element2 and so on. They will follow the same behavior as element0. For readability, they are not
    listed in the table.

    .. list-table:: Vocabulary Behavior under different settings.

        * - vocab_method
          - raw (handle outside)
          - indexing
          - indexing
          - one-hot
          - one-hot
        * - need_pad
          - assume False
          - True
          - False
          - True
          - False
        * - get_pad_value
          - None
          - 0
          - None
          - [0,0,0]
          - None
        * - inner_mapping
          - None
          - 0:pad 1:element0
          - 0:element0
          - -1:<PAD> 0:element0
          - 0:element0
        * - element2repr
          - raise Error
          - pad->0 element0->1
          - element0->0
          - <PAD>->[0,0,0] element0->[1,0,0]
          - element0->[1,0,0]
        * - id2element
          - raise Error
          - 0->pad 1->element0
          - 0->element0
          - -1 -> <PAD> 0->element0 (be careful)
          - 0->element0

    Args:
        method (str): The method to represent element in vocabulary.
        need_pad (bool): Whether to add <PAD> element in vocabulary.
        use_unk (bool): Whether to add <UNK> element in vocabulary.
            Elements that are not found in vocabulary will be directed
            to <UNK> element.

    Attributes:
        method (str): Same as above.
        need_pad (bool): Same as above.
        use_unk (bool): Same as above.
        next_id (int): The id that will be used when next element is added.
        element2id_dict (Dict): This stores the mapping from element to id.
        id2element_dict (Dict): This stores the mapping from id to element.
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

    @property
    def state(self) -> Dict:
        return {
            "method": self.method,
            "need_pad": self.need_pad,
            "use_unk": self.use_unk,
            "element2id_dict": self.element2id_dict,
            "id2element_dict": self.id2element_dict,
            "next_id": self.next_id
        }

    @classmethod
    def from_state(cls, state: Dict) -> object:
        obj = cls(state["method"], state["need_pad"], state["use_unk"])
        obj.element2id_dict = state["element2id_dict"]
        obj.id2element_dict = state["id2element_dict"]
        obj.next_id = state["next_id"]
        return obj

    def add_element(self, element: Hashable):
        r"""This function will add element to the vocabulary.

        Args:
            element (Hashable): The element to be added.
        """
        if element not in self.element2id_dict:
            self.element2id_dict[element] = self.next_id
            self.id2element_dict[self.next_id] = element
            self.next_id += 1

    def id2element(self, idx: int) -> Hashable:
        r"""This function will map id to element.

        Args:
            idx (int): The queried id of element.

        Returns:
            Hashable: The corresponding element if exist.
                Check the behavior of this function
                under different setting in the documentation.

        Raises:
            KeyError: If the id is not found.
        """
        return self.id2element_dict[idx]

    def element2repr(self, element: Hashable) \
                    -> Union[int, List[int]]:
        r"""This function will map element to representation.

        Args:
            element (Hashable): The queried element.

        Returns:
            Union[int, List[int]]: The corresponding representation
                of the element. Check the behavior of this function
                under different setting in the documentation.

        Raises:
            KeyError: If element is not found and vocabulary does
                not use <UNK> element.
        """
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
        r"""This function return the size of vocabulary.

        Returns:
            int: The number of elements, including
                <PAD>, <UNK>.
        """
        return len(self.element2id_dict)

    def has_element(self, element: Hashable) -> bool:
        r"""This function checks whether an element is added to vocabulary.

        Args:
            element (Hashable): The queried element.

        Returns:
            bool: Whether element is found.
        """
        return element in self.element2id_dict

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        r"""This function will loop over the (element, id) pair.

        Returns:
            Iterable[Tuple[Hashable, int]]: (element, id) pair.
        """
        return self.element2id_dict.items()

    def get_dict(self) -> Dict[Hashable, int]:
        r"""This function will get the inner mapping from element to id.

        Returns:
            Dict: The maintained mapping from element to id.
        """
        return self.element2id_dict

    def get_pad_value(self) -> Union[None, int, List[int]]:
        r"""This function will get the PAD element for the vocabulary.

        Returns:
            Union[None, int, List[int]]: The PAD element. Check
                the behavior of this function in the documentation.
        """
        if self.need_pad:
            return self.element2repr(self.PAD_ELEMENT)
        return None
