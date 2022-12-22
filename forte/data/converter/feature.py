#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
from copy import deepcopy
from typing import List, Any, Tuple, Union, Dict, Optional

from forte.common import ValidationError
from forte.data.vocabulary import Vocabulary

__all__ = ["Feature"]


class Feature:
    """
    This class represents a type of feature for a single data instance. The
    Feature can be multiple dimensions. It has methods to do padding and
    retrieve the actual multi-dimension data.

    Args:
        data: A list of features, where each feature can be the value or
            another list of features. Typically this should be the output from
            :meth:`extract` in
            :class:`~forte.data.base_extractor.BaseExtractor`.

        metadata: A dictionary storing meta-data for this feature.
            Mandatory fields includes: `dim`, `dtype`.

            - `dim` indicates the total number of dimension for this
              feature.
            - `dtype` is the value type. For example, it can be `torch.long`.

        vocab: An optional fields about the
            :class:`~forte.data.vocabulary.Vocabulary` used to build this
            feature.

    Please refer to :meth:`data` for the typical usage of this class.
    """

    def __init__(
        self, data: List, metadata: Dict, vocab: Optional[Vocabulary] = None
    ):
        self._meta_data: Dict = metadata
        self._validate_metadata()

        # Pad that will be used to do padding.
        self._pad_value: Union[int, List] = self._meta_data["pad_value"]
        self._dim: int = self._meta_data["dim"]
        self._dtype = self._meta_data["dtype"]

        # Indicating whether current Feature is the inner most feature.
        self._leaf_feature: bool = self._dim == 1
        # Only leaf feature has actual `data`
        self._data: Union[None, List] = None
        # Only non-leaf features have `sub_features`. It can be considered as
        # a list of Feature instances
        self._sub_features: List = []
        # The elements of mask will indicate whether the corresponding value
        # in the data is the actual value or padded data. `mask` will only
        # indicate feature along current dimension. Sub-dimension mask will
        # be stored inside sub features in `sub_features` It will be updated
        # when the method `pad` is called.
        self._mask: List = []

        self._vocab: Optional[Vocabulary] = vocab

        self._parse_sub_features(data)
        self._validate_input()

    def _validate_metadata(self):
        necessary_fields = ["dim", "dtype"]
        for field in necessary_fields:
            if field not in self._meta_data:
                raise ValidationError(f"Field not found in metadata: {field}")

        if self.need_pad and "pad_value" not in self._meta_data:
            raise ValidationError("need_pad is True but no pad_value is given")

    def _validate_input(self):
        """
        Validate input parameters based on some pre-conditions.
        """
        if self.leaf_feature and self._data is None:
            raise ValidationError("Leaf feature should contain actual data.")
        if not self.leaf_feature and self._data:
            raise ValidationError(
                "Non-leaf feature should not contain actual data."
            )
        if not self.leaf_feature and not self._sub_features:
            raise ValidationError(
                "Non-leaf feature should contain sub features."
            )
        if self._dim < 1:
            raise ValidationError("The `dim` in meta should be at least 1.")

    def _parse_sub_features(self, data):
        """
        If current feature is the leaf feature, store the input data. Otherwise,
        parse the data into sub features represented as a list of features.
        Meanwhile, update the mask list.
        Args:
            data (List):
                A list of features, where each feature can be the value or
                another list of features.
        """
        if self.leaf_feature:
            self._data = data
        else:
            self._sub_features = []
            for sub_data in data:
                sub_metadata = deepcopy(self._meta_data)
                sub_metadata["dim"] = sub_metadata["dim"] - 1
                self._sub_features.append(
                    Feature(
                        data=sub_data, metadata=sub_metadata, vocab=self._vocab
                    )
                )

        self._mask = [1] * len(data)

    @property
    def leaf_feature(self) -> bool:
        """
        Returns:
            True if current feature is leaf feature. Otherwise, False.
        """
        return self._leaf_feature

    @property
    def dtype(self):
        """
        Returns:
            The data type of this feature.
        """
        return self._dtype

    @property
    def sub_features(self) -> List["Feature"]:
        """
        Returns:
            A list of sub features. Raise exception if current feature is the
            leaf feature.
        """
        if self._leaf_feature:
            raise ValidationError("Leaf feature does not have sub features")
        if self._dim <= 1:
            raise ValidationError(
                "Non-leaf feature should have as least 2 dimension"
            )

        return self._sub_features

    @property
    def meta_data(self) -> Dict:
        """
        Returns:
            A `Dict` of meta data describing this feature.
        """
        return self._meta_data

    @property
    def vocab(self) -> Optional[Vocabulary]:
        """
        Returns:
            The :class:`~forte.data.vocabulary.Vocabulary` used to build this
            feature.
        """
        return self._vocab

    @property
    def dim(self) -> int:
        """
        Returns:
            The dimension of this feature.
        """
        return self._dim

    @property
    def need_pad(self) -> bool:
        """
        Returns:
            Whether the Feature need to be padded.
        """
        return (
            self._meta_data["need_pad"]
            if "need_pad" in self._meta_data
            else True
        )

    def __len__(self):
        if self.leaf_feature:
            if self._data is None:
                raise ValueError(
                    "Invalid internal state: leaf_feature "
                    "does not have actual data"
                )
            else:
                return len(self._data)
        else:
            return len(self._sub_features)

    def pad(self, max_len: int):
        """
        Pad the current feature dimension with the given `max_len`. It will use
        `pad_value` to do the padding.

        Args:
            max_len (int):
                The padded length.
        """
        if len(self) > max_len:
            raise ValidationError(
                "Feature length should not exceed given max_len"
            )

        for _ in range(max_len - len(self)):
            if self.leaf_feature:
                if self._data is None:
                    raise ValueError(
                        "Invalid internal state: leaf_feature "
                        "does not have actual data"
                    )
                self._data.append(self._pad_value)
            else:
                sub_metadata = deepcopy(self._meta_data)
                sub_metadata["dim"] = sub_metadata["dim"] - 1
                self._sub_features.append(
                    Feature(data=[], metadata=sub_metadata, vocab=self._vocab)
                )
            self._mask.append(0)

    @property
    def data(self) -> Tuple[List[Any], List[Any]]:
        """
        It will return the actual data stored. Internally, it will recursively
        retrieve data from inner dimension features. Meanwhile, it will also
        return a list of masks representing the mask along different dimensions.

        Returns:
            A `Tuple` where

            The first element is the actual data representing this feature.

            The second element is a list of masks. masks[i] in this list
            represents the mask along i-th dimension.

        Here are some examples for how the padding works:

        Example 1 (1-dim feature, no padding):

        .. code-block:: python

            data = [2,7,8]
            meta_data = {
                "pad_value": 0
                "dim": 1
                "dtype": torch.long
            }
            feature = Feature(data, meta_data=meta_data)

            data, masks = feature.data

            # data is:
            # [2,7,8]

            # masks is:
            # [
            #   [1,1,1]
            # ]

        Example 2 (1-dim feature, scalar padding):

        .. code-block:: python

            data = [2,7,8]
            meta_data = {
                "pad_value": 0
                "dim": 1
                "dtype": torch.long
            }
            feature = Feature(data, meta_data=meta_data)

            feature.pad(max_len=4)

            data, masks = feature.data

            # data is:
            # [2,7,8,0]

            # masks is:
            # [
            #   [1,1,1,0]
            # ]

        Example 3 (2-dim feature, scalar padding):

        .. code-block:: python

            data = [[1,2,5], [3], [1,5]]
            meta_data = {
                "pad_value": 0
                "dim": 2
                "dtype": torch.long
            }
            feature = Feature(data, meta_data=meta_data)

            feature.pad(max_len=4)
            for sub_feature in feature.sub_features:
                sub_feature.pad(max_len=3)

            data, masks = feature.data

            # data is:
            # [[1,2,5], [3,0,0], [1,5,0], [0,0,0]]

            # masks is:
            # [
            #   [1,1,1,0],
            #   [[1,1,1], [1,0,0], [1,1,0], [0,0,0]]
            # ]

        Example 4 (1-dim feature, vector padding):

        .. code-block:: python

            data = [[0,1,0],[1,0,0]]
            meta_data = {
                "pad_value": [0,0,1]
                "dim": 1
                "dtype": torch.long
            }
            feature = Feature(data, meta_data=meta_data)

            feature.pad(max_len=3)

            data, masks = feature.data

            # data is:
            # [[0,1,0], [1,0,0], [0,0,1]]

            # masks is:
            # [
            #  [1,1,0]
            # ]
        """
        if self.leaf_feature:
            if self._data is None:
                raise ValueError(
                    "Invalid internal state: leaf_feature "
                    "does not have actual data"
                )
            return self._data, [self._mask]
        else:
            unroll_features: List = []
            sub_stack_masks: List = []

            for feature in self.sub_features:
                sub_unroll_features, sub_masks = feature.data

                for i in range(self._dim - 1):
                    if i == len(sub_stack_masks):
                        sub_stack_masks.append([])
                    sub_stack_masks[i].append(sub_masks[i])

                unroll_features.append(sub_unroll_features)

            return unroll_features, [self._mask] + sub_stack_masks
