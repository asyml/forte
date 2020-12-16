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
A model that copies the parameter states of a pytorch nn module
and performs parameter updates locally.
"""

import copy
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import texar.torch as tx


__all__ = [
    "MetaModule",
    "TexarBertMetaModule"
]


class MetaModule(nn.ModuleList):
    # pylint: disable=line-too-long
    r"""A memory-efficient model that registers the parameters of a
    :class:`torch.nn.Module` and performs parameter updates locally.

    This code is adapted from:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/magic_module.py

    It implements the calculation:
    :math:`L(\theta - \nabla_{\theta} L_{train}(\theta, \phi))`.

    Args:
        module: A :class:`torch.nn.Module`.

    In order to perform :meth:`forward` the same way as the input module,
    we need to copy into this class the helper functions that are called by the
    nested :meth:`forward` of the input module.

    For example, if the input module is tx.modules.BERTClassifier,
    :meth:`_get_noise_shape`, :meth:`_split_heads`, :meth:`_combine_heads`
    are needed to be exposed in this class, so that this :meth:`forward` can
    recognize these functions that are used by the nested :meth:`forward` in
    tx.modules.embedders and tx.modules.encoders.
    """

    def __init__(self, module: nn.Module):
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

    def update_params(self, deltas: Dict[str, torch.Tensor]):
        sub_params: Dict[str, torch.Tensor] = {}
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


class TexarBertMetaModule(MetaModule,
                          tx.modules.EmbedderBase,
                          tx.modules.MultiheadAttentionEncoder):

    def __init__(self, module: nn.Module):
        MetaModule.__init__(module)

    def forward(self, *args, **kwargs):
        return MetaModule.forward(*args, **kwargs)

    # # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/embedders/embedder_base.py#L63
    # def _get_noise_shape(self, dropout_strategy: str,
    #                      ids_rank: Optional[int] = None,
    #                      dropout_input: Optional[torch.Tensor] = None) \
    #         -> Optional[Tuple[int, ...]]:
    #     # pylint: disable=line-too-long
    #
    #     if dropout_strategy == 'element':
    #         noise_shape = None
    #     elif dropout_strategy == 'item':
    #         assert dropout_input is not None
    #         assert ids_rank is not None
    #         shape_a = dropout_input.size()[:ids_rank]
    #         shape_b = (1,) * self._dim_rank     # type: ignore
    #         noise_shape = shape_a + shape_b
    #     elif dropout_strategy == 'item_type':
    #         noise_shape = (self._num_embeds,) + (1,) * self._dim_rank   # type: ignore
    #     else:
    #         raise ValueError(f"Unknown dropout strategy: {dropout_strategy}")
    #     return noise_shape
    #
    # # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/encoders/multihead_attention.py#L232
    # def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
    #     r"""Split channels (dimension 2) into multiple heads,
    #     becomes dimension 1). Must ensure ``x.shape[-1]`` can be
    #     divided by num_heads.
    #     """
    #     depth = x.size(-1)
    #     split_x = torch.reshape(x, (
    #         x.size(0), x.size(1),
    #         self._hparams.num_heads, depth // self._hparams.num_heads))
    #     return split_x.permute((0, 2, 1, 3))
    #
    # # https://github.com/asyml/texar-pytorch/blob/master/texar/torch/modules/encoders/multihead_attention.py#L243
    # def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
    #     r"""
    #
    #     Args:
    #         x: A Tensor of shape ``[batch, num_heads, seq_len, dim]``
    #     Returns:
    #         A Tensor of shape ``[batch, seq_len, num_heads * dim]``
    #     """
    #     t = x.permute((0, 2, 1, 3))  # [batch, seq_len, num_heads, dim]
    #     num_heads, dim = t.size()[-2:]
    #     assert num_heads == self._hparams.num_heads
    #     return torch.reshape(t, (t.size(0), t.size(1), num_heads * dim))
