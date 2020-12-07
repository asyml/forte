# Copyright 2020 The Forte Authors. All Rights Reserved.
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
"""
A model that copies some necessary states of a pytorch module
and performs parameter updates.
"""

import copy
from typing import Optional, Tuple
import operator
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


__all__ = [
    "MetaModule"
]


class MetaModule(nn.Module):
    # pylint: disable=line-too-long
    r"""A model that copies some necessary states of a pytorch module
    and performs parameter updates.

    There is an example code for this class here:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/magic_module.py

    It implements the calculation:
        L(theta - \nabla_{theta} L_{train}(theta, phi)).

    In order to perform :meth:`forward` the same way as the input module,
    we need to copy some of the necessary functions that are needed during the runtime
    of the nested `forward` into this class.
    For example, if the input module is tx.modules.BERTClassifier,
    :meth:`_get_noise_shape`, :meth:`_split_heads`, :meth:`_combine_heads` are needed
    to be exposed in this class.
    """

    def __init__(self, module):
        r"""
        Args:
            module: A pytorch module.
        """

        nn.Module.__init__(self)
        self._type = type(module)

        for key, value in module._parameters.items():
            if value is not None:
                self.register_parameter('_origin_' + key, value)
                self.register_buffer(key, value.data)
            else:
                self.register_buffer(key, None)

        for key, value in module._buffers.items():
            self.register_buffer(key, copy.deepcopy(value))

        for key, value in module._modules.items():
            self.add_module(key, MetaModule(value))

        for key, value in module.__dict__.items():
            if key not in self.__dict__ and\
                    key not in self._buffers and\
                    key not in self._modules:
                self.__setattr__(key, value)

    def forward(self, *args, **kwargs):
        return self._type.forward(self, *args, **kwargs)

    def update_params(self, deltas):
        sub_params = {}
        for key, delta in deltas.items():
            if '.' not in key:
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split('.')[0]
                if attr not in sub_params:
                    sub_params[attr] = {}
                sub_params[attr]['.'.join(key.split('.')[1:])] = delta
        for key, value in sub_params.items():
            self._modules[key].update_params(value)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __len__(self):
        return len(self._modules)

    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]

    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py#L150
    def _get_abs_string_index(self, idx):
        # pylint: disable=C0325
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/embedders/embedder_base.py#L63
    def _get_noise_shape(self, dropout_strategy: str,
                         ids_rank: Optional[int] = None,
                         dropout_input: Optional[torch.Tensor] = None) \
            -> Optional[Tuple[int, ...]]:
        # pylint: disable=line-too-long

        if dropout_strategy == 'element':
            noise_shape = None
        elif dropout_strategy == 'item':
            assert dropout_input is not None
            assert ids_rank is not None
            shape_a = dropout_input.size()[:ids_rank]
            shape_b = (1,) * self._dim_rank     # type: ignore
            noise_shape = shape_a + shape_b
        elif dropout_strategy == 'item_type':
            noise_shape = (self._num_embeds,) + (1,) * self._dim_rank   # type: ignore
        else:
            raise ValueError(f"Unknown dropout strategy: {dropout_strategy}")
        return noise_shape

    # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/encoders/multihead_attention.py#L232
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""Split channels (dimension 2) into multiple heads,
        becomes dimension 1). Must ensure ``x.shape[-1]`` can be
        divided by num_heads.
        """
        depth = x.size(-1)
        split_x = torch.reshape(x, (
            x.size(0), x.size(1),
            self._hparams.num_heads, depth // self._hparams.num_heads))
        return split_x.permute((0, 2, 1, 3))

    # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/encoders/multihead_attention.py#L243
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x: A Tensor of shape ``[batch, num_heads, seq_len, dim]``
        Returns:
            A Tensor of shape ``[batch, seq_len, num_heads * dim]``
        """
        t = x.permute((0, 2, 1, 3))  # [batch, seq_len, num_heads, dim]
        num_heads, dim = t.size()[-2:]
        assert num_heads == self._hparams.num_heads
        return torch.reshape(t, (t.size(0), t.size(1), num_heads * dim))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def conv2d_forward(self, input, weight):
        assert issubclass(self._type, nn.Conv2d)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv2d(nn.functional.pad(
                input, expanded_padding, mode='circular'),
                weight, self.bias, self.stride,
                _pair(0), self.dilation, self.groups)
        return nn.functional.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _check_input_dim(self, input):
        assert issubclass(self._type, nn.BatchNorm2d)
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
