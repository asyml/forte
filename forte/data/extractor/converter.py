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
from typing import List, Tuple, Any

import torch
from torch import Tensor

from forte.data.extractor.feature import Feature


class Converter:
    def __init__(self, need_pad=True):
        self.need_pad = need_pad

    def convert(self, features: List[Feature]) -> \
            Tuple[Tensor, Tensor]:
        # Padding the features if needed
        if self.need_pad:
            # BFS to pad each dimension
            queue: List[Feature] = []
            curr_max_len: int = -1

            for feature in features:
                queue.append(feature)
                curr_max_len = max(curr_max_len, feature.get_len())

            while queue:
                size: int = len(queue)
                next_max_len = -1

                while size > 0:
                    feature: Feature = queue.pop(0)

                    feature.pad(curr_max_len)

                    if not feature.is_base_feature:
                        for sub_feature in feature.get_sub_features():
                            next_max_len = max(next_max_len,
                                               sub_feature.get_len())
                            queue.append(sub_feature)

                    size -= 1

                curr_max_len = next_max_len

        # Convert features to tensors
        padded_features: List[List[Any]] = []
        masks: List[List[Any]] = []
        for feature in features:
            padded_feature, mask = feature.unroll()
            padded_features.append(padded_feature)
            masks.append(mask)

        return torch.tensor(padded_features), torch.tensor(masks)
