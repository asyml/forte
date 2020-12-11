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
from typing import List, Tuple, Any, Optional
import torch
from torch import Tensor

from forte.common import ValidationError
from forte.data.converter.feature import Feature


class Converter:
    """
    This class has the functionality of converting a batch of
    :class:`forte.data.converter.Feature` to a PyTorch `Tensor`. It will also
    do the padding for the given batch of :class:`forte.data.converter.Feature`.

    Args:
        default_dtype: the `dtype` of resulted tensor that will be obtained via
            converting the input features.
    """

    def __init__(self, default_dtype: torch.dtype = torch.long):
        self._default_dtype = default_dtype

    def convert(self, features: List[Feature]) -> \
            Tuple[Tensor, List[Tensor]]:
        """
        Convert a list of Features to a Tensor. Internally it will use
        breadth-first search to pad all features and get the actual data and
        corresponding masks.

        Args:
            features (List[Feature]):
                A list of :class:`forte.data.converter.Feature`

        Returns:
            A `Tuple` containing two elements.

            The first element is a `Tensor` representing the padded batch of
            data. The shape will be
            `(feature_num, feature_dim1_max, feature_dim2_max, ...)`

            The second element is a `List` of `Tensor` representing masks
            along different feature dimensions. For example, the masks[i] is the
            mask `Tensor` along ith dimension and will have the shape:
            `(feature_num, feature_dim1_max, ..., feature_dimi_max)`

            Where `feature_num` equals to ``len(features)``.

        Example 1:

        .. code-block:: python

            data = [[1,2,3], [4,5], [6,7,8,9]]
            meta_data = {
                "pad_value": 0
                "dim": 1
                "dtype": torch.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter()

            tensor, masks = converter.convert(features)

            # tensor is:
            # torch.tensor([[1,2,3,0], [4,5,0,0], [6,7,8,9]], dtype=torch.long)

            # masks is:
            # [
            #     torch.tensor([[1,1,1,0], [1,1,0,0], [1,1,1,1]],
            #                  dtype=torch.bool)
            # ]

        Example 2:

        .. code-block:: python

            data = [[[1,2,3], [4,5]], [[3]]]
            meta_data = {
                "pad_value": 0
                "dim": 2
                "dtype": torch.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter()

            tensor, masks = converter.convert(features)

            # tensor is:
            # torch.tensor([[[1,2,3], [4,5,0]],
            #               [[3,0,0], [0,0,0]]], dtype=torch.long)

            # masks is:
            # [
            #     torch.tensor([[1,1], [1,0]], dtype=torch.bool),
            #     torch.tensor([[[1,1,1], [1,1,0]],
            #                   [[1,0,0], [0,0,0]]], dtype=torch.bool)
            # ]
        """
        dtype: Optional[torch.dtype] = None

        # BFS to pad each dimension
        queue: List[Feature] = []
        curr_max_len: int = -1

        for feature in features:
            if not dtype:
                dtype = feature.dtype
            else:
                if dtype != feature.dtype:
                    raise ValidationError(
                        "The dtype should be same within a batch of Features")
            queue.append(feature)
            curr_max_len = max(curr_max_len, len(feature))

        while len(queue) > 0:
            size: int = len(queue)
            next_max_len = -1
            while size > 0:
                feature: Feature = queue.pop(0)
                feature.pad(curr_max_len)

                if not feature.leaf_feature:
                    for sub_feature in feature.sub_features:
                        next_max_len = max(next_max_len,
                                           len(sub_feature))
                        queue.append(sub_feature)

                size -= 1

            curr_max_len = next_max_len

        # Convert features to tensors
        batch_padded_features: List[List[Any]] = []
        batch_masks: List[List[Any]] = []
        for feature in features:
            padded_feature, mask_list = feature.data
            batch_padded_features.append(padded_feature)
            batch_masks.append(mask_list)
        batch_padded_features_tensor: Tensor = \
            torch.tensor(batch_padded_features, dtype=dtype)

        batch_masks_tensor_list: List[Tensor] = []
        for i in range(features[0]._dim):
            curr_dim_masks = []
            for mask in batch_masks:
                curr_dim_masks.append(mask[i])
            batch_masks_tensor_list.append(torch.tensor(curr_dim_masks,
                                                        dtype=torch.bool))

        return batch_padded_features_tensor, batch_masks_tensor_list
