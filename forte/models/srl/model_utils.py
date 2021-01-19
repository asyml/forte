# Copyright 2019 The Forte Authors. All Rights Reserved.
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

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import texar.torch as tx

LSTMState = Tuple[torch.Tensor, torch.Tensor]


class CustomLSTMCell(tx.core.RNNCellBase[LSTMState]):
    # pylint: disable=super-init-not-called
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        # skip super class constructor
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called

        self._input_size = input_size
        self._hidden_size = hidden_size

        self._dropout_rate = dropout
        self._dropout_mask: Optional[torch.Tensor] = None

        self._initial_cell = nn.Parameter(torch.Tensor(1, hidden_size))
        self._initial_hidden = nn.Parameter(torch.Tensor(1, hidden_size))
        self._projection = nn.Linear(input_size + hidden_size, 3 * hidden_size)

        self.reset_parameters()

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @staticmethod
    @torch.no_grad()
    def orthonormal_init(param: nn.Parameter, n_blocks: int):
        size0, size1 = param.size()
        size0 //= n_blocks
        size_min = min(size0, size1)
        init_values = []
        for _ in range(n_blocks):
            m1 = torch.randn(size0, size0, dtype=param.dtype)
            m2 = torch.randn(size1, size1, dtype=param.dtype)
            q1, r1 = torch.qr(m1)
            q2, r2 = torch.qr(m2)
            q1 *= torch.sign(torch.diag(r1))
            q2 *= torch.sign(torch.diag(r2))
            value = torch.mm(q1[:, :size_min], q2[:size_min, :])
            init_values.append(value)
        param.data = torch.cat(init_values, dim=0)

    def reset_parameters(self):
        self.orthonormal_init(self._projection.weight, 3)
        nn.init.xavier_normal_(self._initial_cell)
        nn.init.xavier_normal_(self._initial_hidden)

    def init_batch(self):
        self._dropout_mask = None

    def zero_state(self, batch_size: int) -> LSTMState:
        return (self._initial_hidden.expand(batch_size, -1),
                self._initial_cell.expand(batch_size, -1))

    def forward(
            self, input_tensor: torch.Tensor, state: Optional[LSTMState] = None
    ) -> Tuple[torch.Tensor, LSTMState]:
        batch_size = input_tensor.size(0)
        if state is None:
            state = self.zero_state(batch_size)
        h, c = state
        if self.training and self._dropout_rate > 0.0:
            if self._dropout_mask is None:
                keep_prob = 1 - self._dropout_rate
                self._dropout_mask = input_tensor.new_zeros(
                    batch_size, self._hidden_size).bernoulli_(keep_prob)
            h = h * self._dropout_mask
        concat_proj = self._projection(torch.cat([input_tensor, h], dim=1))
        i, g, o = torch.chunk(concat_proj, 3, dim=1)
        i = torch.sigmoid(i)
        new_c = (1 - i) * c + i * torch.tanh(g)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)
        return new_h, (new_h, new_c)


class CustomBiLSTM(tx.modules.EncoderBase):
    def __init__(self, hparams=None):
        super().__init__(hparams)

        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()
        input_dim = self._hparams.input_dim
        hidden_dim = self._hparams.hidden_dim
        for _ in range(self._hparams.num_layers):
            fw_cell = CustomLSTMCell(input_dim, hidden_dim)
            bw_cell = CustomLSTMCell(input_dim, hidden_dim)
            self.fw_cells.append(fw_cell)
            self.bw_cells.append(bw_cell)
            input_dim = 2 * hidden_dim

        self.highway_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
            for _ in range(self._hparams.num_layers - 1)])

        self.dropout = nn.Dropout(self._hparams.dropout)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "input_dim": 200,
            "hidden_dim": 200,
            "num_layers": 3,
            "dropout": 0.2,
        }

    def forward(self, inputs: torch.Tensor,
                sequence_length: Optional[torch.LongTensor] = None) \
            -> torch.Tensor:
        for idx in range(self._hparams.num_layers):
            (fw_outputs, bw_outputs), _ = \
                tx.utils.rnn.bidirectional_dynamic_rnn(
                    self.fw_cells[idx], self.bw_cells[idx],
                    inputs, sequence_length)
            outputs = torch.cat([fw_outputs, bw_outputs], dim=2)
            outputs = self.dropout(outputs)
            if idx > 0:
                gate = self.highway_layers[idx - 1](outputs)
                outputs = gate * outputs + (1 - gate) * inputs
            inputs = outputs
        return inputs


class CharCNN(tx.ModuleBase):
    __torch_device__: torch.device

    def __init__(self, char_vocab: tx.data.Vocab, hparams=None):
        super().__init__(hparams)

        self.char_vocab = char_vocab
        self.char_embed = tx.modules.WordEmbedder(
            vocab_size=self.char_vocab.size, hparams={
                "dim": self._hparams.char_embed_size,
            })
        self.cnn_kernels = nn.ModuleList([
            nn.Conv1d(
                self._hparams.char_embed_size,
                self._hparams.filter_size,
                kernel_size=width
            ) for width in self._hparams.filter_widths])
        self._max_filter_width = max(self._hparams.filter_widths)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "char_embed_size": 8,
            "filter_widths": [3, 4, 5],
            "filter_size": 50,
        }

    @property
    def output_size(self):
        return len(self._hparams.filter_widths) * self._hparams.filter_size

    @property
    def _device(self) -> torch.device:
        if not hasattr(self, '__torch_device__'):
            self.__torch_device__ = next(self.parameters()).device
        return self.__torch_device__

    def forward(self, words: List[List[str]]) -> torch.Tensor:
        r"""Construct character-level representations of the words.

        Args:
            words: A list of list of :class:`str`, representing words of
                each sentence in a batch.

        Returns:
            A tensor of shape `(batch_size, max_length, output_size)`, where

            - `batch_size` is equal to ``len(words)``.
            - `max_length` is equal to ``max(len(s) for s in words)``, i.e.
              length of the longest sentence in the batch.
            - `output_size` is equal to ``self.output_size``.

            For sentences shorter than `max_length`, the extra positions are
            padded with zero vectors.
        """
        sent_lengths = [len(sent) for sent in words]
        max_length = max(sent_lengths)
        all_words = [w for sent in words for w in sent]
        max_word_len = max(len(w) for w in all_words)
        char_indices = np.zeros((len(all_words), max_word_len), dtype=np.int64)
        for idx, word in enumerate(all_words):
            char_ids = self.char_vocab.map_tokens_to_ids_py(list(word))
            char_indices[idx, :len(word)] = char_ids
        indices = torch.from_numpy(char_indices).to(self._device)

        # embed: (batch_size * max_length, char_embed_dim, max_word_len)
        embed = self.char_embed(indices).transpose(1, 2)
        if max_word_len < self._max_filter_width:
            pad_length = self._max_filter_width - max_word_len
            embed = torch.cat([embed, embed.new_zeros(
                *embed.size()[:2], pad_length)], dim=2)
        kernel_outputs = [kernel(embed) for kernel in self.cnn_kernels]
        cnn_output = torch.cat(
            [torch.max(out, dim=2)[0] for out in kernel_outputs], dim=1)

        sent_cnn_outputs = torch.split(cnn_output, sent_lengths, dim=0)
        output = cnn_output.new_zeros(
            len(words), max_length, cnn_output.size(1))
        for idx, sent_output in enumerate(sent_cnn_outputs):
            output[idx, :sent_lengths[idx]] = sent_output
        return output


def sum_list(xs: List[torch.Tensor]) -> torch.Tensor:
    if len(xs) == 1:
        return xs[0]
    mid = len(xs) // 2
    return sum_list(xs[:mid]) + sum_list(xs[mid:])


# pylint: disable=unused-argument,function-redefined
@overload
def batch_gather(tensors: List[torch.Tensor],
                 index: torch.LongTensor) -> List[torch.Tensor]: ...


@overload
def batch_gather(tensors: torch.Tensor,
                 index: torch.LongTensor) -> torch.Tensor: ...


def batch_gather(tensors, index):
    batch_size, query_size = index.size()
    value_size = None

    def _get_value_size_fn(tensor):
        nonlocal value_size
        if value_size is not None or not isinstance(tensor, torch.Tensor):
            return
        value_size = tensor.size(1)

    tx.utils.map_structure(_get_value_size_fn, tensors)
    # assert torch.max(index).item() < value_size
    offsets = torch.arange(batch_size, device=index.device)
    offsets = (offsets * value_size).repeat_interleave(query_size)
    index = index.contiguous().view(-1) + offsets

    def _gather_fn(tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        tensor = tensor.contiguous().view(-1, *tensor.size()[2:])
        values = torch.index_select(tensor, dim=0, index=index)
        values = values.view(batch_size, -1, *values.size()[1:])
        return values

    gathered_tensors = tx.utils.map_structure(_gather_fn, tensors)
    return gathered_tensors


# pylint: enable=unused-argument,function-redefined


class MLP(tx.ModuleBase):
    def __init__(self, hparams=None):
        super().__init__(hparams)

        input_size = self._hparams.input_size
        layers = []
        dropout = self._hparams.dropout_rate
        if self._hparams.activation is not None:
            activation = tx.core.get_layer({"type": self._hparams.activation})
        else:
            activation = None
        for has_bias, output_size in zip(
                self._hparams.has_bias, self._hparams.hidden_size):
            layers.append(nn.Linear(input_size, output_size, bias=has_bias))
            input_size = output_size
            if activation is not None:
                layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout, inplace=True))
        output_size = self._hparams.output_size
        layers.append(nn.Linear(input_size, output_size, bias=False))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "input_size": 300,
            "num_layers": 2,
            "has_bias": [True, True],
            "hidden_size": [150, 150],
            "activation": None,
            "output_size": 1,
            "dropout_rate": 0.0,
        }

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(input_tensor)


class ConcatInputMLP(tx.ModuleBase):
    _cached_inputs: List[torch.Tensor]

    def __init__(self, hparams=None):
        super().__init__(hparams)

        mlp_hparams = hparams.copy()
        mlp_hparams["input_size"] = sum(self._hparams.input_sizes)
        del mlp_hparams["input_sizes"]
        self.mlp = MLP(mlp_hparams)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "input_sizes": [150, 150],
            "num_layers": 2,
            "has_bias": [True, True],
            "hidden_size": [150, 150],
            "output_size": 1,
            "dropout_rate": 0.0,
            "activation": None,
        }

    @contextmanager
    def cache_results(self, inputs: List[torch.Tensor]):
        self._cached_inputs = inputs
        yield
        del self._cached_inputs

    def forward(self, inputs: List[Union[torch.Tensor,
                                         List[Tuple[torch.Tensor,
                                                    torch.Tensor]]]]) \
            -> torch.Tensor:
        """

        Args:
            inputs: List of 2D or 3D Tensors (or list of `(index, float)`
                tuples), representing indices for each concat'd part of the
                input tensor. First dimension is the batch dimension.

                - If an element of the list is another list, each inner list
                  element must be a tuple of `(index, coef)`, where `index` is
                  used for indexing, and all indexed tensors will be summed up,
                  weighted by the tensor `coef`.
                - If the tensor is 2D, it must be a LongTensor.
                - If the tensor is 3D, it must be a FloatTensor representing the
                  soft-IDs (weights for weighted average).

        Returns:
        """
        parts = []
        for index, cache in zip(inputs, self._cached_inputs):
            if isinstance(index, list):
                raise NotImplementedError
            if index.dim() == 3:
                if cache.dim() == 2:
                    # [batch, query_size, value_size] x [value_size, dim]
                    # Make it 3D so we can perform batched MM.
                    cache = cache.expand(index.size(0), *cache.size())
                # [batch, query_size, value_size] x [batch, value_size, dim]
                values = torch.bmm(index, cache)
            elif cache.dim() == 3:
                # Batch-specific values. Add batch offset to indices for lookup.
                values = batch_gather(cache, index)
            else:
                values = F.embedding(index, cache)
            parts.append(values)
        output = self.mlp(torch.cat(parts, dim=-1))
        return output
