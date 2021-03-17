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
import logging
from typing import List, Tuple, Dict, Union, Hashable, Iterable

import texar.torch as tx

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

    Note that these two special tokens are necessary for the system in certain
    cases and thus must present in the vocabulary. The behavior of these
    special tokens are pre-defined based on different settings. To get around
    the default behavior (for example, if you have a pre-defined vocabulary
    with different setups), you can instruct the class to not adding these
    tokens automatically, and use the :func:`mark_special_element` instead.

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
        pad_value (int): Value used by <PAD> element. Default is 0.
        unk_value (int): Value used by <UNK> element. Default is 1.

    Attributes:
        method (str): Same as above.
        use_pad (bool): Same as above.
        use_unk (bool): Same as above.
        pad_value (int): Same as above.
        unk_value (int): Same as above.
        next_id (int): The id that will be used when next element is added.
        element2id_dict (dict): This stores the mapping from element to id.
        id2element_dict (dict): This stores the mapping from id to element.
    """

    # TODO: It is better to add a generic type for this class, now every element
    #   is simply empty.
    def __init__(self, method: str, need_pad: bool, use_unk: bool,
                 pad_value: int = 0, unk_value: int = 1):
        self.method = method
        self.need_pad = need_pad
        self.use_unk = use_unk

        Note: most of the time, you don't have to call this method yourself,
        but should let the `init` function to handle that.

        self.next_id = 0
        if method == "one-hot" and need_pad:
            pad_value = -1
            logging.warning(
                "Cannot use 0 as pad id if one-hot method is used. "
                "Chaning pad id to -1!")
        if need_pad:
            self.add_special_element(Vocabulary.PAD_ELEMENT, pad_value)

        if use_unk:
            self.add_special_element(Vocabulary.UNK_ELEMENT, unk_value)

    def add_special_element(self, element: Hashable, element_id: int):
        r"""
        Add special_elements, such as PAD and UNK.
        Args:
            element (Hashable): The element to be added.
            element_id (int): The id assigned to the element.
        """
        if element_id in self.id2element_dict:
            raise ValueError(
                    "ID %d is alreay being used in Vocabulary. " % element_id)
        if element not in self.element2id_dict:
            self.element2id_dict[element] = element_id
            self.id2element_dict[element_id] = element

    def add_element(self, element: ElementType, representation: Any = None,
                    count: int = 1) -> int:
        r"""This function will add a regular element to the vocabulary.

        Args:
            element (Hashable): The element to be added.
            representation: The vocabulary representation of this element
                will use this value. For example, you may want to use `-100`
                for ignored tokens for PyTorch skipped tokens. Note that the
                class do not check whether this representation is used by
                another element, so the caller have to manage the behavior
                itself.
            count (int): the count to be incremented for this element, default
                is 1 (i.e. consider it appear once on every add). This value
                will have effect only if `do_counting` is True.

        Returns:
            The internal id of the element.
        """
        while self.next_id in self.id2element_dict:
            self.next_id += 1
        if element not in self.element2id_dict:
            self.element2id_dict[element] = self.next_id
            self.id2element_dict[self.next_id] = element
            self.next_id += 1
            eid = self.next_id
        return eid

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
            self, element: Union[ElementType, str]) -> Union[int, List[int]]:
        r"""This function will map element to representation.

        Args:
            element (Hashable): The queried element. It can be either the same
              type as the element, or string (for the special tokens).

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
                element, self._element2id[self._base_special_tokens['UNK']])
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
        """Compute the one-hot encoding on the fly."""
        vec_size = len(self._element2id)
        if self.use_pad:
            vec_size -= 1
        vec = [0 for _ in range(vec_size)]
        if idx != -1:
            vec[idx] = 1
        return vec

    def __len__(self) -> int:
        r"""This function return the size of vocabulary.

        Returns:
            int: The number of elements, including <PAD>, <UNK>.
        """
        return len(self._element2id)

    def has_element(self, element: Union[ElementType, str]) -> bool:
        r"""This function checks whether an element is added to vocabulary.

        Args:
            element (Hashable): The queried element.

        Returns:
            bool: Whether element is found.
        """
        return element in self._element2id

    def vocab_items(self) -> Iterable[Tuple[Union[ElementType, str], int]]:
        r"""This function will loop over the (element, id) pair inside this
        class.

        Returns:
            Iterable[Tuple]: Iterables of (element, id) pair.
        """
        return self._element2id.items()

    def get_pad_value(self) -> Union[None, int, List[int]]:
        r"""This function will get the representation of the PAD element for
        the vocabulary. The representation depends on the settings of this
        class, it can be an integer or a list of int (e.g. a vector).

        Returns:
            Union[None, int, List[int]]: The PAD element. Check
            the behavior of this function in the class documentation.
        """
        if self.use_pad:
            return self.element2repr(self._base_special_tokens['PAD'])
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
        # Make a new vocab class:
        # 1. We do not add the PAD or UNK at init, but will copy them later.
        # 2. We also ignore other special tokens at init, but copy them later.
        # 2. We follow the do_counting setup.
        vocab: Vocabulary = Vocabulary(
            self.method, use_pad=False, use_unk=False,
            do_counting=self.do_counting
        )
        # We then set these flag manually based on this vocabulary.
        vocab.use_pad = self.use_pad
        vocab.use_unk = self.use_unk

        # Now we copy all the vocabulary items to the new vocab.
        for element, eid in self.vocab_items():
            # Copy the special tokens regardless of the filter.
            if self.is_special_token(eid):
                element_name = None
                try:
                    if element == self._base_special_tokens["PAD"]:
                        element_name = "PAD"
                except KeyError:
                    # No PAD in the origin vocab.
                    pass

                try:
                    if element == self._base_special_tokens["UNK"]:
                        element_name = "UNK"
                except KeyError:
                    # No UNK in the origin vocab.
                    pass

                # Special element value must be string.
                assert isinstance(element, str)
                vocab.add_special_element(
                    element, eid, self._id2repr.get(element, None),
                    element_name)
            elif not vocab_filter.filter(eid):
                vocab.add_element(
                    element,
                    count=self.get_count(eid) if self.do_counting else 1
                )
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

        if 0 <= self.max_freq < freq:
            will_filter = True

        return will_filter
