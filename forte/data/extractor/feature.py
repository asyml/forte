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
from typing import List, Any


class Feature:
    def __init__(self, data: List[Any], dim: int, pad_id: int = 0):
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
        self.data: List[Any] = data
        self.dim: int = dim
        self.pad_id: int = pad_id
        self.is_base_feature = dim == 1

    def isBaseFeature(self) -> bool:
        pass

    def getSubFeatures(self) -> List['Feature']:
        pass

    def getLen(self) -> int:
        pass

    def pad(self, max_len: int):
        pass

    def unroll(self) -> List[Any]:
        pass
