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
from typing import List, Any, Tuple, Union


class Feature:
    """
    This class represents a type of feature for a single data instance. The
    Feature can be multiple dimensions. It has methods to do padding and
    retrieve the actual multi-dimension data.
    The `data` contains the actual value. The `pad_id` is the pad that
    will be used to do padding. The `dim` indicates the total number of
    dimension for this feature.
    Here are some examples for how the padding works:
    i) [2,7,8] -> [2,7,8,0]
        1 dim feature
        pad_len = 4
        pad_id = 0
    ii) [[1,2],[3,4,5],[9]] -> [[1,2,0],[3,4,5],[9,0,0],[0,0,0]]
        2 dim feature
        pad_len_dim1 = 4, pad_len_dim2=3
        pad_id = 0
    iii) [[0,1,0],[1,0,0]] -> [[0,1,0],[1,0,0],[0,0,1]]
        1 dim one-hot-encoding feature
        pad_len = 3
        pad_id = [0,0,1]
    """
    def __init__(self,
                 data: List,
                 pad_id: Union[int, List],
                 dim: int):
        """
        Args:
            data (List):
                A list of features, where each feature can be the value or
                another list of features.
            pad_id (int or List):
                a single integer or a list of integer representing <PAD>. Only
                the base dimension will actually use `pad_id`.
            dim (int):
                Total number of dimensions for the data. `dim` is always >= 1.
                If the data is a list of value, the `dim` should be 1 and this
                feature is called base feature.
        """
        self.pad_id: Union[int, List] = pad_id
        self.dim: int = dim

        # Indicating whether current Feature is the inner most feature.
        self._is_base_feature: bool = dim == 1
        # Only base feature has actual `data`
        self.data: Union[None, List] = None
        # Only non-base features have `sub_features`. It can be considered as
        # a list of Feature instances
        self.sub_features: List = []
        # The elements of mask will indicate whether the corresponding value
        # in the data is the actual value or padded data. `mask` will only
        # indicate feature along current dimension. Sub-dimension mask will
        # be stored inside sub features in `sub_features` It will be updated
        # when the method `pad` is called.
        self.mask: List = []

        self._parse_sub_features(data)
        self._validate_input()

    def _validate_input(self):
        """
        Validate input parameters based on some pre-conditions.
        """
        assert (self.is_base_feature and self.data is not None or
                (not self.is_base_feature
                 and self.data is None
                 and self.sub_features is not None))
        assert type(self.pad_id) == int or type(self.pad_id) == list
        assert self.dim >= 1

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
        if self.is_base_feature:
            self.data: List = data
        else:
            self.sub_features: List = []
            for sub_data in data:
                self.sub_features.append(
                    Feature(sub_data, self.pad_id, self.dim - 1))

        self.mask: List = [1] * len(data)

    @property
    def is_base_feature(self) -> bool:
        """
        Return whether or not the current feature is the base feature.
        Returns: True if current feature is base feature. Otherwise, False.
        """
        return self._is_base_feature

    def get_sub_features(self) -> List['Feature']:
        """
        Retrieve a list of sub features. The call is valid only when current
        dimension is not the base dimension.
        Returns: a list of sub features.
        """
        assert not self.is_base_feature, \
            "Base feature does not have sub features"
        assert self.dim > 1, \
            "Non-base feature should have as least 2 dimension"

        return self.sub_features

    def __len__(self):
        return len(self.data) if self.is_base_feature else \
            len(self.sub_features)

    def pad(self, max_len: int):
        """
        Pad the current feature dimension with the given `max_len`. It will use
        `pad_id` to do the padding.
        Args:
            max_len (int):
                The padded length.
        """
        assert len(self) <= max_len, \
            "Feature length should not exceed given max_len"

        for i in range(max_len - len(self)):
            if self.is_base_feature:
                self.data.append(self.pad_id)
            else:
                self.sub_features.append(Feature([], self.pad_id, self.dim - 1))
            self.mask.append(0)

    def unroll(self, need_pad: bool = True) -> Tuple[List[Any], List[Any]]:
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
        if not need_pad:
            return self.data, []

        if self.is_base_feature:
            return self.data, [self.mask]
        else:
            unroll_features: List = []
            sub_stack_masks: List = []

            for feature in self.get_sub_features():
                sub_unroll_features, sub_masks = feature.unroll()

                for i in range(self.dim - 1):
                    if i == len(sub_stack_masks):
                        sub_stack_masks.append([])
                    sub_stack_masks[i].append(sub_masks[i])

                unroll_features.append(sub_unroll_features)

            return unroll_features, [self.mask] + sub_stack_masks
