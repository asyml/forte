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
        self.data: List[Any] = data
        self.pad_id: Union[int, List] = pad_id
        self.dim: int = dim
        self.base_feature = dim == 1
        self.mask = [1 * len(data)]

        self.validate_input()

    def validate_input(self):
        assert type(self.data) == list
        assert type(self.pad_id) == int or type(self.pad_id) == list
        assert self.dim >= 1

    def is_base_feature(self) -> bool:
        return self.base_feature

    def get_sub_features(self) -> List['Feature']:
        assert not self.is_base_feature(), \
            "Base feature does not have sub features"
        assert self.dim > 1, \
            "Non-base feature should have as least 2 dimension"

        features: List = []
        for sub_data in self.data:
            if type(sub_data) == list:
                # Normal data
                features.append(
                    Feature(sub_data, self.pad_id, self.dim - 1))
            elif type(sub_data) == Feature:
                # Padded data
                features.append(sub_data)
            else:
                raise ValueError("Unexpected sub feature type: " +
                                 type(sub_data))

        return features

    def get_len(self) -> int:
        return len(self.data)

    def pad(self, max_len: int):
        assert self.get_len() <= max_len, \
            "Feature length should not exceed given max_len"

        for i in range(max_len - self.get_len()):
            self.data.append(Feature([], self.pad_id, self.dim - 1))
            self.mask.append(0)

    def unroll(self, need_pad: bool = True) -> Tuple[List[Any], List[Any]]:
        if not need_pad:
            return self.data, []

        unroll_features: List = []

        if self.is_base_feature:
            for data in self.data:
                if type(data) != Feature:
                    # Actual value
                    unroll_features.append(self.pad)
                else:
                    # Padded data
                    unroll_features.append(data)

            return unroll_features, self.mask
        else:
            total_sub_masks: List = []

            for feature in self.get_sub_features():
                sub_unroll_features, sub_masks = feature.unroll()
                unroll_features.append(sub_unroll_features)
                total_sub_masks.append(sub_masks)

            return unroll_features, self.mask + total_sub_masks

