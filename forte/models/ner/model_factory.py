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

from typing import Dict
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import texar.torch as texar
from texar.torch.modules.embedders import WordEmbedder

from forte.common.configuration import Config
from forte.models.ner.conditional_random_field import ConditionalRandomField


class BiRecurrentConvCRF(nn.Module):
    def __init__(
        self,
        word_vocab: Dict,
        char_vocab_size: int,
        tag_vocab_size: int,
        config_model: Config,
    ):
        super().__init__()

        self.word_embedder = WordEmbedder(
            init_value=texar.data.Embedding(
                vocab=word_vocab,
                hparams={
                    "dim": config_model.word_emb.dim,
                    "file": config_model.embedding_path,
                    "read_fn": "load_glove",
                },
            ).word_vecs
        )

        self.char_embedder = WordEmbedder(
            vocab_size=char_vocab_size, hparams=config_model.char_emb
        )

        self.char_cnn = torch.nn.Conv1d(**config_model.char_cnn_conv)

        self.dropout_in = nn.Dropout2d(config_model.dropout_rate)
        # standard dropout
        self.dropout_rnn_in = nn.Dropout(config_model.dropout_rate)
        self.dropout_out = nn.Dropout(config_model.dropout_rate)

        self.rnn = nn.LSTM(
            config_model.bilstm_sentence_encoder.rnn_cell_fw.input_size,
            config_model.bilstm_sentence_encoder.rnn_cell_fw.kwargs.num_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dense = nn.Linear(
            config_model.bilstm_sentence_encoder.rnn_cell_fw.kwargs.num_units
            * 2,
            config_model.output_hidden_size,
        )

        self.tag_projection_layer = nn.Linear(
            config_model.output_hidden_size, tag_vocab_size
        )

        self.crf = ConditionalRandomField(
            tag_vocab_size, constraints=None, include_start_end_transitions=True
        )

        if config_model.initializer is None or callable(
            config_model.initializer
        ):
            self.initializer = config_model.initializer
        else:
            self.initializer = texar.core.layers.get_initializer(
                config_model["initializer"]
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return

        for name, parameter in self.named_parameters():
            if name.find("embedder") == -1 and name.find("crf") == -1:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.0)
                else:
                    self.initializer(parameter)

    def forward(self, input_word, input_char, target=None, mask=None, hx=None):
        """

        Args:
            input_word:
            input_char:
            target:
            mask:
            hx:

        Returns: the loss value

        """
        output, _, mask, _ = self.encode(input_word, input_char, mask, hx)

        logits = self.tag_projection_layer(output)
        log_likelihood = (
            self.crf.forward(logits, target, mask) / target.size()[0]
        )
        return -log_likelihood

    def decode(self, input_word, input_char, mask=None, hx=None):
        """
        Args:
            input_word:
            input_char:
            mask:
            hx:

        Returns:

        """
        output, _, mask, _ = self.encode(
            input_word, input_char, mask=mask, hx=hx
        )

        logits = self.tag_projection_layer(output)
        best_paths = self.crf.viterbi_tags(logits, mask.long())
        predicted_tags = [x for x, y in best_paths]
        predicted_tags = [torch.tensor(x).unsqueeze(0) for x in predicted_tags]
        predicted_tags = texar.utils.pad_and_concat(
            predicted_tags, axis=0, pad_constant_values=0
        )

        return predicted_tags

    def encode(self, input_word, input_char, mask=None, hx=None):
        # output from rnn [batch, length, tag_space]
        length = mask.sum(dim=1).long()

        # [batch, length, word_dim]
        word = self.word_embedder(input_word)
        word = self.dropout_in(word)

        # [batch, length, char_length, char_dim]
        char = self.char_embedder(input_char)
        char_size = char.size()
        # first transform to [batch * length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(
            char_size[0] * char_size[1], char_size[2], char_size[3]
        ).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.char_cnn(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)

        # independently apply dropout to word and characters
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([word, char], dim=2)
        input = self.dropout_rnn_in(input)

        # prepare packed_sequence
        seq_input, hx, rev_order, mask = prepare_rnn_seq(
            input, length, hx=hx, masks=mask, batch_first=True
        )
        self.rnn.flatten_parameters()
        seq_output, hn = self.rnn(seq_input, hx=hx)
        output, hn = recover_rnn_seq(
            seq_output, rev_order, hx=hn, batch_first=True
        )

        # apply dropout for the output of rnn
        output = self.dropout_out(output)

        # [batch, length, tag_space]
        output = self.dropout_out(F.elu(self.dense(output)))

        return output, hn, mask, length


def prepare_rnn_seq(rnn_input, lengths, hx=None, masks=None, batch_first=False):
    """

    Args:
        rnn_input: [seq_len, batch, input_size]:
            tensor containing the features of the input sequence.
        lengths: [batch]:
            tensor containing the lengthes of the input sequence
        hx: [num_layers * num_directions, batch, hidden_size]:
            tensor containing the initial hidden state for each element
            in the batch.
        masks: [seq_len, batch]:
            tensor containing the mask for each element in the batch.
        batch_first:
            If True, then the input and output tensors are provided as
            [batch, seq_len, feature].

    Returns:

    """

    def check_decreasing(lengths):
        lens, order = torch.sort(lengths, dim=0, descending=True)
        if torch.ne(lens, lengths).sum() == 0:
            return None
        else:
            _, rev_order = torch.sort(order)
            return lens, order, rev_order

    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
        rev_order = None
    else:
        lens, order, rev_order = check_res
        batch_dim = 0 if batch_first else 1
        rnn_input = rnn_input.index_select(batch_dim, order)
        if hx is not None:
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, order)
                cx = cx.index_select(1, order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, order)

    lens = lens.tolist()
    seq = rnn_utils.pack_padded_sequence(
        rnn_input, lens, batch_first=batch_first
    )

    if masks is not None:
        if batch_first:
            masks = masks[:, : lens[0]]
        else:
            masks = masks[: lens[0]]

    return seq, hx, rev_order, masks


def recover_rnn_seq(seq, rev_order, hx=None, batch_first=False):
    output, _ = rnn_utils.pad_packed_sequence(seq, batch_first=batch_first)
    if rev_order is not None:
        batch_dim = 0 if batch_first else 1
        output = output.index_select(batch_dim, rev_order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, rev_order)
                cx = cx.index_select(1, rev_order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, rev_order)
    return output, hx
