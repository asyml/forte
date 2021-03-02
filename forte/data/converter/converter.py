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
import logging
from typing import List, Tuple, Any, Optional, Union, Dict
import numpy as np
import torch

from forte.common.configuration import Config
from forte.common import ValidationError
from forte.data.converter.feature import Feature

logger = logging.getLogger(__name__)

__all__ = [
    "Converter"
]


class Converter:
    """
    This class has the functionality of converting a batch of
    :class:`~forte.data.converter.Feature` to a PyTorch `Tensor`. It can also
    do the padding for the given batch of :class:`~forte.data.converter.Feature`
    if user requested it. Please refer to the `request` parameter in
    :class:`~forte.train_preprocessor.TrainPreprocessor` for details.

    Args:
        config: An instance of `Dict` or
            :class:`~forte.common.configuration.Config` that provides all
            configurable options. See :meth:`default_configs` for available
            options and default values.
    """

    def __init__(self, config: Union[Dict, Config]):
        self._config = Config(config,
                              default_hparams=self.default_configs(),
                              allow_new_hparam=True)

    @staticmethod
    def default_configs():
        """
        Returns a dictionary of default hyper-parameters.

        .. code-block:: python

            {
                "to_numpy": True,
                "to_torch": True
            }

        Here:

            `"to_numpy"`: bool
                Whether convert to `numpy.ndarray`.
                Default is True.

            `"to_torch"`: bool
                 Whether convert to `torch.tensor`. Default is True.

            .. note::
                If `need_pad` in :class:`forte.data.converter.Feature`
                is False and `to_numpy` and `to_torch` is True,
                it will raise an exception if the target data cannot be
                converted to `numpy.ndarray` or `torch.tensor`.

        .. note::
            If `need_pad` in :class:`forte.data.converter.Feature`
            is True and `to_torch` is True, `to_torch` will overwrite the
            effect of `to_numpy`.
        """
        return {
            "to_numpy": True,
            "to_torch": True
        }

    @property
    def to_numpy(self) -> bool:
        return self._config.to_numpy

    @property
    def to_torch(self) -> bool:
        return self._config.to_torch

    @property
    def state(self) -> Dict:
        return {
            "to_numpy": self.to_numpy,
            "to_torch": self.to_torch
        }

    def load_state(self, state: Dict):
        self._config.to_numpy = state["to_numpy"]
        self._config.to_torch = state["to_torch"]

    def convert(self, features: List[Feature]) -> \
            Tuple[Any, List[Any]]:
        """
        Convert a list of Features to actual data, where

        1. The outer most dimension will always be the batch dimension (i.e
        `len(output) = len(feature_num)`).

        2. The type can be:

            2.1 A `List` of primitive `int` or another `List`

            2.2 A `numpy.ndarray`

            2.3 A `torch.Tensor`

        If `need_pad` in :class:`forte.data.converter.Feature` is True, it
        will pad all features with given `pad_value` stored inside
        :class:`forte.data.converter.Feature`.

        If `to_numpy` is True, it will try to convert data into
        `numpy.ndarray`.

        If `to_torch` is True, it will try to convert data into
        `torch.tensor`.

        Args:
            features (List[Feature]):
                A list of :class:`forte.data.converter.Feature`

        Returns:
            A `Tuple` containing two elements.

            1. The first element is either a `List` or `numpy.ndarray` or
            `torch.tensor` representing the batch of data.

            2. The second element is a `List` or `numpy.ndarray` representing
            masks along different feature dimensions.

        Example 1:

        .. code-block:: python

            data = [[1,2,3], [4,5], [6,7,8,9]]
            meta_data = {
                "pad_value": 0,
                "need_pad": True,
                "dim": 1
                "dtype": np.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter(to_numpy=True,
                                  to_torch=False)

            output_data, masks = converter.convert(features)

            # output_data is:
            # np.array([[1,2,3,0], [4,5,0,0], [6,7,8,9]], dtype=np.long)

            # masks is:
            # [
            #     np.array([[1,1,1,0], [1,1,0,0], [1,1,1,1]],
            #              dtype=np.bool)
            # ]

        Example 2:

        .. code-block:: python

            data = [[[1,2,3], [4,5]], [[3]]]
            meta_data = {
                "pad_value": 0,
                "need_pad": True,
                "dim": 2
                "dtype": np.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter(to_numpy=True,
                                  to_torch=False)

            output_data, masks = converter.convert(features)

            # output_data is:
            # np.array([[[1,2,3], [4,5,0]], [[3,0,0], [0,0,0]]],
            #          dtype=np.long)


            # masks is:
            # [
            #     np.array([[1,1], [1,0]], dtype=np.bool),
            #     np.array([[[1,1,1], [1,1,0]],
            #              [[1,0,0], [0,0,0]]], dtype=np.bool)
            # ]

        Example 3:

        .. code-block:: python

            data = [[1,2,3,0], [4,5,0,0], [6,7,8,9]]
            meta_data = {
                "pad_value": 0
                "need_pad": False,
                "dim": 1
                "dtype": np.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter(need_pad=False)

            output_data, _ = converter.convert(features)

            # output_data is:
            # torch.tensor([[1,2,3,0], [4,5,0,0], [6,7,8,9]], dtype=torch.long)

        Example 4:

        .. code-block:: python

            data = [[1,2,3], [4,5], [6,7,8,9]]
            meta_data = {
                "pad_value": 0,
                "need_pad": True,
                "dim": 1
                "dtype": np.long
            }
            features = [Feature(i, meta_data=meta_data) for i in data]
            converter = Converter(to_torch=True)

            output_data, masks = converter.convert(features)

            # output_data is:
            # torch.tensor([[1,2,3,0], [4,5,0,0], [6,7,8,9]], dtype=torch.long)

            # masks is:
            # [
            #     torch.tensor([[1,1,1,0], [1,1,0,0], [1,1,1,1]],
            #                  dtype=np.bool)
            # ]
        """
        dtype: Optional[np.dtype] = None

        need_pad: bool = features[0].need_pad

        if need_pad and self.to_torch and not self.to_numpy:
            logger.warning("need_pad is True and to_torch is True, "
                           "setting to_numpy to False will be ignored.")

        # Do padding if needed
        if need_pad:
            dtype = self._padding(features)

        # Collect a batch of data & masks from Features
        data_list: List[List[Any]] = []
        # batch_masks_per_example:
        # (feature_num, feature_dim, feature_mask1, [feature_mask2, ...])
        masks_per_example_list: List[List[Any]] = []
        for feature in features:
            padded_feature, mask_list = feature.data
            data_list.append(padded_feature)
            masks_per_example_list.append(mask_list)

        # Switch the two outer most dimensions
        # batch_list:
        # (feature_dim, feature_num, feature_mask1, [feature_mask2, ...])
        masks_list: List[List[Any]] = []
        for i in range(features[0].dim):
            curr_dim_masks = []
            for mask in masks_per_example_list:
                curr_dim_masks.append(mask[i])
            masks_list.append(curr_dim_masks)

        # Convert to target type
        if not self.to_numpy and not self.to_torch:
            return data_list, masks_list

        # Note: to_torch == True overwrite to_numpy option
        if self.to_torch:
            data_tensor: torch.Tensor = \
                self._to_tensor_type(data_list, dtype)
            masks_tensor_list: List[torch.Tensor] = []
            for batch_masks_dim_i in masks_list:
                masks_tensor_list.append(
                    self._to_tensor_type(batch_masks_dim_i, np.bool))

            return data_tensor, masks_tensor_list

        if self.to_numpy:
            data_np: np.ndarray = \
                self._to_numpy_type(data_list, dtype)
            masks_np_list: List[np.ndarray] = []
            for batch_masks_dim_i in masks_list:
                masks_np_list.append(
                    self._to_numpy_type(batch_masks_dim_i, np.bool))

            return data_np, masks_np_list

        # Control should not reach here
        raise RuntimeError("Invalid converter internal state")

    @staticmethod
    def _padding(features: List[Feature]) -> Optional[torch.dtype]:
        # BFS to pad each dimension
        queue: List[Feature] = []
        curr_max_len: int = -1
        dtype: Optional[torch.dtype] = None

        for feature in features:
            if not dtype:
                dtype = feature.dtype
            else:
                if dtype != feature.dtype:
                    raise ValidationError(
                        "The dtype should be same within a batch of Features")
            if not feature.need_pad:
                raise ValidationError(
                    "Inconsistent need pad flag for a batch of Features")
            queue.append(feature)
            curr_max_len = max(curr_max_len, len(feature))

        while len(queue) > 0:
            size: int = len(queue)
            next_max_len = -1
            while size > 0:
                feature = queue.pop(0)
                feature.pad(curr_max_len)

                if not feature.leaf_feature:
                    for sub_feature in feature.sub_features:
                        next_max_len = max(next_max_len,
                                           len(sub_feature))
                        queue.append(sub_feature)

                size -= 1

            curr_max_len = next_max_len

        return dtype

    @staticmethod
    def _to_numpy_type(data: List[Any], dtype) -> np.ndarray:
        return np.array(data, dtype=dtype)

    @staticmethod
    def _to_tensor_type(data: List[Any], dtype) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype)
