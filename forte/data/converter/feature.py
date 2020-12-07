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
import torch

from forte.data.extractor.vocabulary import Vocabulary


class Feature:
    """
    This class represents a type of feature for a single data instance. The
    Feature can be multiple dimensions. It has methods to do padding and
    retrieve the actual multi-dimension data.
    Here are some examples for how the padding works:
    i) [2,7,8] -> [2,7,8,0]
        1 dim feature
        pad_len = 4
        pad_value = 0
    ii) [[1,2],[3,4,5],[9]] -> [[1,2,0],[3,4,5],[9,0,0],[0,0,0]]
        2 dim feature
        pad_len_dim1 = 4, pad_len_dim2=3
        pad_value = 0
    iii) [[0,1,0],[1,0,0]] -> [[0,1,0],[1,0,0],[0,0,1]]
        1 dim one-hot-encoding feature
        pad_len = 3
        pad_value = [0,0,1]
    A typical usage of this class will be like the following:
    .. code-block:: python
                # create a feature
                feature = Feature(data=[[1,2],[3,4,5],[9]],
                                  {
                                    "pad_value": 0,
                                    "dim": 2,
                                    "dtype": torch.long
                                  })
                # Pad current dim with max_len=4
                feature.pad(4)

                # Pad each 2nd dim (the base dim) with max_len=3
                for sub_feature in feature.get_sub_features():
                    sub_feature.pad(3)

                # Retrieve the actual data
                data = feature.unroll()

                # The data is a list of list which looks like the following:
                # data = [[1,2,0],[3,4,5],[9,0,0],[0,0,0]]
                ...
    """
    def __init__(self,
                 data: List,
                 metadata: Dict,
                 vocab: Optional[Vocabulary] = None):
        """
        Args:
            data (List):
                A list of features, where each feature can be the value or
                another list of features.
                The `data` contains the actual value.
            metadata(Dict):
                A dictionary of metadata for this feature. Mandatory metadata
                fields includes: `pad_value`, `dim`, `dtype`.
                The `pad_value` is the pad that will be used to do padding.
                The `dim` indicates the total number of dimension for this
                feature.
                The `dtype` is the value type. For example, it can be
                torch.long.
            vocab(Vocabulary):
                An optional fields about the vocabulary used to build this
                feature.
        """
        self._meta_data: Dict = metadata
        self._validate_metadata()

        self._pad_value: Union[int, List] = self._meta_data["pad_value"]
        self._dim: int = self._meta_data["dim"]
        self._dtype = self._meta_data["dtype"]

        # Indicating whether current Feature is the inner most feature.
        self._base_feature: bool = self._dim == 1
        # Only base feature has actual `data`
        self._data: Union[None, List] = None
        # Only non-base features have `sub_features`. It can be considered as
        # a list of Feature instances
        self._sub_features: List = []
        # The elements of mask will indicate whether the corresponding value
        # in the data is the actual value or padded data. `mask` will only
        # indicate feature along current dimension. Sub-dimension mask will
        # be stored inside sub features in `sub_features` It will be updated
        # when the method `pad` is called.
        self._mask: List = []

        self._vocab: Vocabulary = vocab

        self._parse_sub_features(data)
        self._validate_input()

    def _validate_metadata(self):
        necessary_fields = ["pad_value", "dim", "dtype"]
        for field in necessary_fields:
            assert field in self._meta_data, \
                "Field not found in metadata: {}".format(field)

    def _validate_input(self):
        """
        Validate input parameters based on some pre-conditions.
        """
        assert (self.base_feature and self._data is not None or
                (not self.base_feature
                 and self._data is None
                 and self._sub_features is not None))
        assert type(self._pad_value) == int or type(self._pad_value) == list
        assert self._dim >= 1

    def _parse_sub_features(self, data):
        """
        If current feature is the base feature, store the input data. Otherwise,
        parse the data into sub features represented as a list of features.
        Meanwhile, update the mask list.
        Args:
            data (List):
                A list of features, where each feature can be the value or
                another list of features.
        """
        if self.base_feature:
            self._data: List = data
        else:
            self._sub_features: List = []
            for sub_data in data:
                sub_metadata = deepcopy(self._meta_data)
                sub_metadata["dim"] = sub_metadata["dim"] - 1
                self._sub_features.append(
                    Feature(data=sub_data,
                            metadata=sub_metadata,
                            vocab=self._vocab))

        self._mask: List = [1] * len(data)

    @property
    def base_feature(self) -> bool:
        """
        Return whether or not the current feature is the base feature.
        Returns: True if current feature is base feature. Otherwise, False.
        """
        return self._base_feature

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns: the data type of this feature
        """
        return self._dtype

    @property
    def data(self) -> List:
        assert self.base_feature, \
            "Non-base feature does not have data"

        return self._data

    @property
    def sub_features(self) -> List['Feature']:
        """
        Retrieve a list of sub features. The call is valid only when current
        dimension is not the base dimension.
        Returns: a list of sub features.
        """
        assert not self.base_feature, \
            "Base feature does not have sub features"
        assert self._dim > 1, \
            "Non-base feature should have as least 2 dimension"

        return self._sub_features

    @property
    def meta_data(self) -> Dict:
        return self._meta_data

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def __len__(self):
        return len(self._data) if self.base_feature else \
            len(self._sub_features)

    def pad(self, max_len: int):
        """
        Pad the current feature dimension with the given `max_len`. It will use
        `pad_value` to do the padding.
        Args:
            max_len (int):
                The padded length.
        """
        assert len(self) <= max_len, \
            "Feature length should not exceed given max_len"

        for i in range(max_len - len(self)):
            if self.base_feature:
                self._data.append(self._pad_value)
            else:
                sub_metadata = deepcopy(self._meta_data)
                sub_metadata["dim"] = sub_metadata["dim"] - 1
                self._sub_features.append(Feature(data=[],
                                                  metadata=sub_metadata,
                                                  vocab=self._vocab))
            self._mask.append(0)

    def unroll(self) -> Tuple[List[Any], List[Any]]:
        """
        It will return the actual data stored. Internally, it will recursively
        retrieve data from inner dimension features. Meanwhile, it will also
        return a list of masks representing the mask along different dimension.
        Args:
            need_pad (bool):
                Indicate whether or not the internal data need to be padded.
        Returns: A tuple where the first element is the actual data and the
        second element is a list of masks. masks[i] in this list represents the
        mask along ith dimension.
        Here are some examples:
        i) unroll([2,7,8,0]) will return:
            ([2,7,8,0], # data
             [1,1,1,0]) # mask
        ii) unroll([[1,2,0],[3,4,5],[9,0,0],[0,0,0]]) will return:
            ([[1,2,0],[3,4,5],[9,0,0],[0,0,0]],                 # data
             [[1,1,1,0],[[1,1,0],[1,1,1],[1,0,0,],[0,0,0]]])    # mask
        iii) unroll([[0,1,0],[1,0,0]]) will return:
            ([[0,1,0],[1,0,0],[0,0,1]], # data
             [1,1,0])                   # mask
        """
        if self.base_feature:
            return self._data, [self._mask]
        else:
            unroll_features: List = []
            sub_stack_masks: List = []

            for feature in self.sub_features:
                sub_unroll_features, sub_masks = feature.unroll()

                for i in range(self._dim - 1):
                    if i == len(sub_stack_masks):
                        sub_stack_masks.append([])
                    sub_stack_masks[i].append(sub_masks[i])

                unroll_features.append(sub_unroll_features)

            return unroll_features, [self._mask] + sub_stack_masks
