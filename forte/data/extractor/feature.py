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
from typing import List, Any, Tuple


class Feature:
    def __init__(self,
                 data: List[Any],
                 pad_id: List[int],
                 dim: int):
        """
        Args:
        data (List[Any]):
            A list of features, where each feature can be the value or another
            list of features.
        dim (int):
            Total number of dimensions for the data. `dim` is always >= 1.
            If the data is a list of value, the `dim` should be 1 and this
            feature is called base feature.
        pad_id (int):
            The id for <PAD> token
        """
        # self.validate_input(data, pad_id, dim)

        self.data: List[Any] = data
        self.pad_id: List[int] = pad_id
        self.dim: int = dim
        self.is_base_feature = dim == 1
        # self.is_pad = is_pad

    def validate_input(self, data: List[int], pad_id: List[int], dim: int):
        assert dim >= 1
        assert len(pad_id) == dim

        data_dim = data
        while dim > 1:
            assert type(data_dim) == list
            assert len(data_dim) > 0
            data_dim = data_dim[0]
            dim -= 1
        assert type(data_dim) == list
        for val in data_dim:
            assert type(val) == float or type(val) == int

    def isBaseFeature(self) -> bool:
        return self.is_base_feature

    def getSubFeatures(self) -> List['Feature']:
        assert not self.is_base_feature, \
            "Base feature does not have sub features"
        assert len(self.pad_id) > 2, \
            "Non-base feature should have pad_id for sub feature"
        assert self.dim > 1, \
            "Non-base feature should have as least 2 dimension"

        features = []
        for sub_data in self.data:
            if type(sub_data) == list:
                # Normal data
                features.append(
                    Feature(sub_data, self.pad_id[1:], self.dim - 1))
            elif type(sub_data) == Feature:
                # Padded data
                features.append(sub_data)
            else:
                raise ValueError("Unexpected sub feature type: " +
                                 type(sub_data))

        return features

    def getLen(self) -> int:
        return len(self.data)

    def pad(self, max_len: int):
        assert self.getLen() <= max_len, \
            "Feature length should not exceed given max_len"

        # for i in range(max_len - self.getLen()):
        #     self.data.append(Feature())

    def unroll(self) -> Tuple[List[Any], List[Any]]:
        pass

#
# class PadFeature(Feature):
#     def __init__(self, data: List[Any], pad_id: List[int], dim: int):
#         super().__init__(data, pad_id, dim)
