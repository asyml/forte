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
from abc import ABC
from collections import Counter
from typing import List, Tuple, Dict, Union, Hashable, Iterable, TypeVar, \
    Generic, Any, Optional, Set

import texar.torch as tx
import logging
from typing import List, Tuple, Dict, Union, Hashable, Iterable, Optional

from forte.common import InvalidOperationException

ElementType = TypeVar('ElementType', bound=Hashable)


class Vocabulary(Generic[ElementType]):
    r"""This class will store "Elements" that are added, assign "Ids" to them
    and return "Representations" if queried. These three are the main concepts
    in this class.

    1. Element: Any hash-able instance that the user want to store.
    2. Id: Each element will have an unique Id, which is an integer.
    3. Representation: according to the configuration, the representation for
       an element could be an integer (in this case, would be "Id"), or
       an one-hot vector (in this case, would be a list of integer).

    The class adopts the special elements from `Texar-Pytorch`, which are:

    1. <PAD>: which will be mapped into Id of 0 or -1 and have
       different representation according to different setting.
    2. <UNK>: if added into the vocabulary,
       will be the default element if the queried element is not found.
    3. <EOS>: End of sentence marker
    4. <BOS>: Begin of sentence marker

    Here is a table on how our Vocabulary class behavior under different
    settings. Element0 means the first element that is added to the vocabulary.
    Elements added later will be element1, element2 and so on. They will follow
    the same behavior as element0. For readability, they are not
    listed in the table.

    .. list-table:: Vocabulary Behavior under different settings.

        * - `vocab_method`
          - custom (handle and implemented by the user)
          - indexing
          - indexing
          - one-hot
          - one-hot
        * - `need_pad`
          - assume False
          - True
          - False
          - True
          - False
        * - `get_pad_value`
          - None
          - 0
          - None
          - [0,0,0]
          - None
        * - `inner_mapping`
          - None
          - 0:pad 1:element0
          - 0:element0
          - -1:<PAD> 0:element0
          - 0:element0
        * - `element2repr`
          - raise Error
          - pad->0 element0->1
          - element0->0
          - <PAD>->[0,0,0] element0->[1,0,0]
          - element0->[1,0,0]
        * - `id2element`
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
        _element2id (dict): This stores the mapping from element to id.
        _id2element (dict): This stores the mapping from id to element.
    """

    # TODO: It is better to add a generic type for this class, now every element
    #   is simply empty.
    def __init__(
            self, method: str = "indexing",
            need_pad: bool = True, use_unk: bool = True,
            add_bos: bool = False, add_eos: bool = False,
            do_counting: bool = True
    ):
        self.method: str = method
        self.need_pad: bool = need_pad
        self.use_unk: bool = use_unk
        self.add_bos: bool = add_bos
        self.add_eos: bool = add_eos
        self.do_counting: bool = do_counting

        self._special_tokens = tx.data.SpecialTokens

        self._pad_id: Optional[int] = None
        self._unk_id: Optional[int] = None
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None

        # Maps the raw element to the internal id.
        self._element2id: Dict = {}
        # Maps the internal id to the raw element.
        self._id2element: Dict = {}
        # Maps the internal id to the representation. This dict is populated
        #  when users provided customized representation of elements.
        self._id2repr: Dict = {}

        # Count the number of appearance of an element, indexed by the element
        #  id.
        self.__counter: Counter = Counter()

        # Keep a set of the special ids.
        self.__special_ids: Set[int] = set()

        if method == "one-hot" and need_pad:
            self.next_id = -1
            logging.warning(
                "Cannot use 0 as pad id if one-hot method is used. "
                "Chaning pad id to -1!")
        else:
            self.next_id = 0

        # When the element type is not string, this will still add these special
        #  tokens.
        if need_pad:
            self.__special_ids.add(self.add_element(self._special_tokens.PAD))

        if use_unk:
            self.__special_ids.add(self.add_element(self._special_tokens.UNK))

        if add_bos:
            self.__special_ids.add(self.add_element(self._special_tokens.BOS))

        if add_eos:
            self.__special_ids.add(self.add_element(self._special_tokens.EOS))

    @property
    def special_ids(self) -> Set[int]:
        """
        Get all the ids of the special tokens.

        Returns: A set containing the ids of the special tokens.

        """
        return self.__special_ids

    def get_ids(self):
        """
        Get all the ids of this vocabulary.

        Returns: All the ids.

        """

        return range(self.__len__())

    def get_count(self, e: Union[ElementType, int]) -> int:
        """
        Get the counts of the vocabulary element.

        Args:
            e: The element to get counts for. It can be the element id or the
              element's raw type.

        Returns:
            The count of the element.
        """
        if not self.do_counting:
            raise InvalidOperationException(
                "The vocabulary is not configured to count the elements.")

        if isinstance(e, int):
            return self.__counter[e]
        else:
            return self.__counter[self._element2id[e]]

    def add_element(self, element: ElementType, representation: Any = None,
                    count: int = 1):
        r"""This function will add element to the vocabulary.

        Args:
            element (Hashable): The element to be added.
            representation: The vocabulary representation of this element
             will use this value. For example, you may want to use `-100`
             for ignored tokens for PyTorch skipped tokens.
            count (int): the count to be incremented for this element, default
             is 1 (i.e. consider it appear once on every add). This value
             will have effect only if `do_counting` is True.

        Returns:
            The internal id of the element.
        """
        element_id: int
        try:
            element_id = self._element2id[element]
            if self.do_counting:
                self.__counter[element_id] += count
        except KeyError:
            element_id = self.next_id
            self._element2id[element] = element_id
            self._id2element[element_id] = element
            if representation:
                self._id2repr[element_id] = representation
            if self.do_counting:
                self.__counter[element_id] = count

            self.next_id += 1

        return element_id

    def id2element(self, idx: int) -> ElementType:
        r"""This function will map id to element.

        Args:
            idx (int): The queried id of element.

        Returns:
            The corresponding element if exist. Check the behavior
             of this function under different setting in the documentation.

        Raises:
            KeyError: If the id is not found.
        """
        return self._id2element[idx]

    def element2repr(
            self, element: ElementType) -> Union[int, List[int]]:
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
            idx = self._element2id.get(
                element, self._element2id[self._special_tokens.UNK])
        else:
            idx = self._element2id[element]

        # If a custom representation is set for this idx, we will use it.
        if idx in self._id2repr:
            return self._id2repr[idx]
        elif self.method == "indexing":
            return idx
        elif self.method == "one-hot":
            return self._one_hot(idx)
        else:
            raise InvalidOperationException(
                f"Cannot find the representation for idx at [{idx}], it does"
                f" not have a customized representation, and the representation"
                f" method [{self.method}] is not supported.")

    def _one_hot(self, idx: int):
        vec_size = len(self._element2id)
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
        return len(self._element2id)

    def has_element(self, element: ElementType) -> bool:
        r"""This function checks whether an element is added to vocabulary.

        Args:
            element (Hashable): The queried element.

        Returns:
            bool: Whether element is found.
        """
        return element in self._element2id

    def items(self) -> Iterable[Tuple[ElementType, int]]:
        r"""This function will loop over the (element, id) pair inside this
        class.

        Returns:
            Iterable[Tuple]: Iterables of (element, id) pair.
        """
        return self._element2id.items()

    def get_pad_value(self) -> Union[None, int, List[int]]:
        r"""This function will get the PAD element for the vocabulary.

        Returns:
            Union[None, int, List[int]]: The PAD element. Check
            the behavior of this function in the documentation.
        """
        if self.need_pad:
            return self.element2repr(self._special_tokens.PAD)
        return None

    def filter(self, vocab_filter: "VocabFilter") -> "Vocabulary":
        """
        This function will create a new vocabulary object, which
        is based on the current vocabulary, but filter out elements that
        appear fewer times than the `min_count` value. Calling this function
        will cause a full iteration over the vocabulary, thus normally, it
        should be called after collecting all the vocabulary in the dataset.

        Args:
            vocab_filter: The filter used to filter the vocabulary.

        Returns:
            A new vocabulary after filtering.
        """
        vocab: Vocabulary = Vocabulary(
            self.method, self.need_pad, self.use_unk,
            self.add_bos, self.add_eos, self.do_counting)

        for eid in self.get_ids():
            # Special ids are added internally.
            if eid in self.__special_ids:
                continue

            # Filtered vocabulary will be ignored.
            if vocab_filter.filter(eid):
                continue

            # Adding vocabulary fulfilling the criteria, along with the count.
            vocab.add_element(
                self._id2element[eid],
                count=self.get_count(eid) if self.do_counting else 1)

        return vocab


class VocabFilter(ABC):
    """
    Base class for vocabulary filters, which is used to implement constraints
    to choose a subset of vocabulary. For example, one can filter out vocab
    elements that happen fewer than a certain frequency.

    Args:
        vocab: The vocabulary object to be filtered.
    """

    def __init__(self, vocab: Vocabulary):
        self._vocab = vocab

    def filter(self, element_id: int) -> bool:
        """
        Given the element id, it will determine whether the element should be
        filtered out.

        Args:
            element_id: The element id to be checked.

        Returns:

        """
        raise NotImplementedError


class FrequencyVocabFilter(VocabFilter):
    """
    A frequency based filter. It will filter vocabulary elements that appear
    fewer than `min_frequency` or more than `max_frequency`. The check will
    be skipped if the threshold values are negative.

    Args:
        vocab: The vocabulary object.
        min_frequency (int): The min frequency threshold, default -1 (i.e. no
          frequency check for min).
        max_frequency (int): The max frequency threhold, default -1 (i.e. no
          frequency check for max).

    """

    def __init__(self, vocab: Vocabulary, min_frequency: int = -1,
                 max_frequency: int = -1):
        super().__init__(vocab)
        self.min_freq = min_frequency
        self.max_freq = max_frequency

        if not vocab.do_counting:
            raise InvalidOperationException(
                "The provided vocabulary is not configured to collect counts, "
                "cannot filter the vocabulary based on counts.")

    def filter(self, element_id: int) -> bool:
        freq = self._vocab.get_count(element_id)

        will_filter = False
        if self.min_freq >= 0 and freq < self.min_freq:
            will_filter = True

        if self.max_freq >= 0 and freq > self.max_freq:
            will_filter = True

        return will_filter
