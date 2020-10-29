#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import List, Any, Tuple, Union


class Feature:
    def __init__(self,
                 data: List[Any],
                 pad_id: Union[int, List],
                 dim: int):
        """
        Args:
        data (List[Any]):
            A list of features, where each feature can be the value or another
            list of features.
        pad_id (int or List):
            a single integer or a nested list of integer representing <PAD>.
        dim (int):
            Total number of dimensions for the data. `dim` is always >= 1.
            If the data is a list of value, the `dim` should be 1 and this
            feature is called base feature.
        """
        self.pad_id: Union[int, List] = pad_id
        self.dim: int = dim
        self.base_feature = dim == 1
        self.data = None  # Only base feature has actual data
        self.mask = None

        self._parse_sub_features(data)
        self._validate_input()

    def _validate_input(self):
        assert (self.is_base_feature() and self.data is not None or
                (not self.is_base_feature()
                 and self.data is None
                 and self.sub_features is not None))
        assert type(self.pad_id) == int or type(self.pad_id) == list
        assert self.dim >= 1

    def _parse_sub_features(self, data):
        if self.is_base_feature():
            self.data = data
        else:
            self.sub_features: List = []
            for sub_data in data:
                self.sub_features.append(
                    Feature(sub_data, self.pad_id, self.dim - 1))

        self.mask = [1] * len(data)

    def is_base_feature(self) -> bool:
        return self.base_feature

    def get_sub_features(self) -> List['Feature']:
        assert not self.is_base_feature(), \
            "Base feature does not have sub features"
        assert self.dim > 1, \
            "Non-base feature should have as least 2 dimension"

        return self.sub_features

    def get_len(self) -> int:
        return len(self.data) if self.is_base_feature() else \
            len(self.sub_features)

    def pad(self, max_len: int):
        assert self.get_len() <= max_len, \
            "Feature length should not exceed given max_len"

        for i in range(max_len - self.get_len()):
            if self.is_base_feature():
                self.data.append(self.pad_id)
            else:
                self.sub_features.append(Feature([], self.pad_id, self.dim - 1))
            self.mask.append(0)

    def unroll(self, need_pad: bool = True) -> Tuple[List[Any], List[Any]]:
        if not need_pad:
            return self.data, []

        if self.is_base_feature():
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
