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
            Tuple[Tensor, List[Tensor]]:
        # Padding the features if needed
        if self.need_pad:
            # BFS to pad each dimension
            queue: List[Feature] = []
            curr_max_len: int = -1

            for feature in features:
                queue.append(feature)
                curr_max_len = max(curr_max_len, feature.get_len())

            while len(queue) > 0:
                size: int = len(queue)
                next_max_len = -1
                while size > 0:
                    feature: Feature = queue.pop(0)
                    feature.pad(curr_max_len)

                    if not feature.is_base_feature():
                        for sub_feature in feature.get_sub_features():
                            next_max_len = max(next_max_len,
                                               sub_feature.get_len())
                            queue.append(sub_feature)

                    size -= 1

                curr_max_len = next_max_len

        # Convert features to tensors
        batch_padded_features: List[List[Any]] = []
        batch_masks: List[List[Any]] = []
        for feature in features:
            padded_feature, mask_list = feature.unroll(self.need_pad)
            batch_padded_features.append(padded_feature)
            batch_masks.append(mask_list)

        stack_masks = []
        for i in range(features[0].dim):
            curr_dim_masks = []
            for mask in batch_masks:
                curr_dim_masks.append(mask[i])
            stack_masks.append(torch.tensor(curr_dim_masks))

        return torch.tensor(batch_padded_features), stack_masks
