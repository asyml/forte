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
__all__ = ["LabeledSpanGraphNetwork"]
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as e:
    raise ImportError(
        " `pytorch` is not installed correctly."
        " Consider install torch "
        "via `pip install torch`."
        " Please refer to [extra requirement for models](pip install forte[models])"
        " for more information. "
    ) from e
from mypy_extensions import TypedDict

try:
    import texar.torch as tx
except ImportError as e:
    raise ImportError(
        " `texar-pytorch` is not installed correctly."
        " Consider install texar via `pip install texar-pytorch`."
        " Or refer to [extra requirement for Texar model support](pip install forte[models])"
        " for more information."
    ) from e

from forte.models.srl import model_utils as utils
from forte.models.srl.data import SRLSpan, Span


class LabeledSpanGraphNetwork(tx.ModuleBase):
    @property
    def output_size(self):
        """
        This module is supposed to be the last layer so we will not return
        an informative output_size
        Returns:

        """
        return 0

    __torch_device__: torch.device

    def __init__(
        self, word_vocab: tx.data.Vocab, char_vocab: tx.data.Vocab, hparams=None
    ):
        super().__init__(hparams)

        # Word vocabulary & representation
        self.word_vocab = word_vocab
        self.word_embed = tx.modules.WordEmbedder(
            init_value=tx.data.Embedding(
                vocab=self.word_vocab.token_to_id_map_py,
                hparams={
                    "file": self._hparams.context_embeddings.path,
                    "dim": self._hparams.context_embeddings.size,
                    "read_fn": "load_glove",
                },
            ).word_vecs
        )
        self.head_embed = tx.modules.WordEmbedder(
            init_value=tx.data.Embedding(
                vocab=self.word_vocab.token_to_id_map_py,
                hparams={
                    "file": self._hparams.head_embeddings.path,
                    "dim": self._hparams.head_embeddings.size,
                    "read_fn": "load_glove",
                },
            ).word_vecs
        )
        self.span_length_embed = tx.modules.PositionEmbedder(
            position_size=self._hparams.max_arg_width,
            hparams={
                "dim": self._hparams.feature_size,
            },
        )

        # Character vocabulary & representation
        self.char_cnn = utils.CharCNN(
            char_vocab=char_vocab,
            hparams={
                "char_embed_size": self._hparams.char_embedding_size,
                "filter_widths": self._hparams.filter_widths,
                "filter_size": self._hparams.filter_size,
            },
        )
        self.embed_dropout = nn.Dropout(self._hparams.lexical_dropout_rate)

        # ELMo representation
        if self._hparams.elmo.path is not None:
            # pylint: disable=import-outside-toplevel
            from allennlp.modules.elmo import Elmo, batch_to_ids

            elmo_hparams = self._hparams.elmo
            self.elmo = Elmo(
                options_file=elmo_hparams.config,
                weight_file=elmo_hparams.path,
                num_output_representations=1,
            )
            self._elmo_char_ids_fn = batch_to_ids
        else:
            self.elmo = None

        # LSTM
        single_hidden_dim = self._hparams.contextualization_size
        lstm_input_dim = self.word_embed.dim + self.char_cnn.output_size
        if self.elmo is not None:
            lstm_input_dim += self._hparams.elmo.dim
        self.lstm = utils.CustomBiLSTM(
            hparams={
                "input_dim": lstm_input_dim,
                "hidden_dim": single_hidden_dim,
                "num_layers": self._hparams.contextualization_layers,
                "dropout": self._hparams.lstm_dropout_rate,
            }
        )
        hidden_dim = single_hidden_dim * 2

        self.label_vocab = {
            label: idx + 1  # reserve index 0 for null label
            for idx, label in enumerate(self._hparams.srl_labels)
        }
        self.label_inverse_vocab = {v: k for k, v in self.label_vocab.items()}
        self.head_attention = nn.Linear(hidden_dim, 1)

        word_input_dim = self.word_embed.dim + self.char_cnn.output_size
        mlp_num_layers = self._hparams.ffnn_depth
        mlp_hparams = {
            "input_sizes": [
                hidden_dim,  # concat'd state at start of span
                hidden_dim,  # concat'd state at end of span
                word_input_dim,
                self.span_length_embed.dim,
            ],
            "num_layers": mlp_num_layers,
            "hidden_size": [self._hparams.ffnn_size] * mlp_num_layers,
            "dropout_rate": self._hparams.dropout_rate,
        }
        self.argument_mlp = utils.ConcatInputMLP(
            hparams={
                **mlp_hparams,
                "output_size": 1,
                "activation": "ReLU",
            }
        )
        self.predicate_mlp = utils.ConcatInputMLP(
            hparams={
                **mlp_hparams,
                "input_sizes": [hidden_dim],
                "output_size": 1,
                "activation": "ReLU",
            }
        )
        self.span_label_mlp = utils.ConcatInputMLP(
            hparams={
                **mlp_hparams,
                "input_sizes": mlp_hparams["input_sizes"] + [hidden_dim],
                "output_size": len(self.label_vocab),
                "activation": "ReLU",
            }
        )

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "filter_widths": [3, 4, 5],
            "filter_size": 50,
            "char_embedding_size": 8,
            "context_embeddings": {
                "path": "embeddings/glove.840B.300d.05.filtered",
                "size": 300,
                "datasets": "txt",
                "lowercase": False,
            },
            "head_embeddings": {
                "path": "embeddings/glove_50_300_2.filtered",
                # "path": "embeddings/glove_50_300_2.txt",
                "size": 300,
                "datasets": "txt",
                "lowercase": False,
            },
            "elmo": {
                "path": None,
                "config": None,
                "dim": 256,
            },
            "contextualizer": "lstm",
            "contextualization_size": 200,
            "contextualization_layers": 3,
            "ffnn_size": 150,
            "ffnn_depth": 2,
            "feature_size": 20,
            "max_span_width": 30,
            "model_heads": True,
            "num_attention_heads": 1,
            "srl_labels": [
                # predicate
                "V",
                # simple propositions
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "AA",
                "AM",
                "AM-ADV",
                "AM-CAU",
                "AM-DIR",
                "AM-DIS",
                "AM-EXT",
                "AM-LOC",
                "AM-MNR",
                "AM-MOD",
                "AM-NEG",
                "AM-PNC",
                "AM-PRD",
                "AM-REC",
                "AM-TM",
                "AM-TMP",
                # propositions with coreferenced arguments
                "C-A0",
                "C-A1",
                "C-A2",
                "C-A3",
                "C-A4",
                "C-A5",
                "C-AM-ADV",
                "C-AM-CAU",
                "C-AM-DIR",
                "C-AM-DIS",
                "C-AM-EXT",
                "C-AM-LOC",
                "C-AM-MNR",
                "C-AM-NEG",
                "C-AM-PNC",
                "C-AM-TMP",
                "C-V",
                # propositions with discontinuous argument
                "R-A0",
                "R-A1",
                "R-A2",
                "R-A3",
                "R-A4",
                "R-AA",
                "R-AM-ADV",
                "R-AM-CAU",
                "R-AM-DIR",
                "R-AM-EXT",
                "R-AM-LOC",
                "R-AM-MNR",
                "R-AM-PNC",
                "R-AM-TMP",
            ],
            "max_arg_width": 30,
            "argument_ratio": 0.8,
            "predicate_ratio": 0.4,
            "lexical_dropout_rate": 0.5,
            "dropout_rate": 0.2,
            "lstm_dropout_rate": 0.4,
        }

    @property
    def _device(self) -> torch.device:
        if not hasattr(self, "__torch_device__"):
            self.__torch_device__ = next(self.parameters()).device
        return self.__torch_device__

    def _create_span_indices(
        self, batch_size: int, max_len: int, max_span: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        start_ids = torch.arange(0, max_len).repeat_interleave(max_span)
        end_ids = start_ids + torch.arange(0, max_span).repeat(max_len)
        valid_mask = end_ids < max_len
        start_ids, end_ids = start_ids[valid_mask], end_ids[valid_mask]
        span_length = end_ids - start_ids
        start_ids = start_ids.expand(batch_size, *start_ids.size()).to(
            device=self._device
        )
        end_ids = end_ids.expand_as(start_ids).to(device=self._device)
        span_length = span_length.expand_as(start_ids).to(device=self._device)
        return start_ids, end_ids, span_length

    @staticmethod
    def _set_submatrix(mat: torch.Tensor, x: int, y: int, value: torch.Tensor):
        mat[x : (x + value.size(0)), y : (y + value.size(1))] = value

    def _create_softmax_mask(
        self, batch_size: int, max_len: int, max_span: int
    ) -> torch.ByteTensor:
        # 1 + 2 + ... + max_span + max_span + ...  (total max_len terms)
        total_lines = (1 + min(max_span, max_len)) * min(max_span, max_len) // 2
        if max_len > max_span:
            total_lines += (max_len - max_span) * max_span

        lower_tri = torch.tril(
            torch.ones(max_span, max_span, dtype=torch.uint8)
        )
        mask = torch.zeros(total_lines, max_len, dtype=torch.uint8)
        line_count = 0
        for idx in range(max_len):
            if max_len - idx < max_span:
                cur_mask = lower_tri[: (max_len - idx), : (max_len - idx)]
            else:
                cur_mask = lower_tri
            self._set_submatrix(mask, line_count, idx, cur_mask)
            line_count += cur_mask.size(0)
        mask = mask.expand(batch_size, total_lines, max_len)
        return mask.to(device=self._device)

    def _filter_labels(
        self,
        start_ids: torch.LongTensor,
        end_ids: torch.LongTensor,
        predicates: torch.LongTensor,
        srls: List[List[SRLSpan]],
    ) -> torch.LongTensor:
        batch_size, num_spans = start_ids.size()
        num_predicates = predicates.size(1)
        device = start_ids.device
        start_ids = start_ids.cpu().numpy()
        end_ids = end_ids.cpu().numpy()
        predicates = predicates.cpu().numpy()
        batch_predicates = [
            {pred: idx for idx, pred in enumerate(preds)}
            for preds in predicates
        ]
        batch_spans = [
            {(l, r): idx for idx, (l, r) in enumerate(zip(starts, ends))}
            for starts, ends in zip(start_ids, end_ids)
        ]

        gold_labels = torch.zeros(
            batch_size, num_predicates * num_spans, dtype=torch.long
        )
        for b_idx in range(batch_size):
            for srl in srls[b_idx]:
                span_idx = batch_spans[b_idx].get((srl.start, srl.end), None)
                predicate_idx = batch_predicates[b_idx].get(srl.predicate, None)
                if span_idx is not None and predicate_idx is not None:
                    label_idx = predicate_idx * num_spans + span_idx
                    gold_labels[b_idx, label_idx] = self.label_vocab[srl.label]
        gold_labels = gold_labels.to(device=device)
        return gold_labels

    def _compute_soft_head_attention_brute(
        self,
        start_ids: torch.LongTensor,
        end_ids: torch.LongTensor,
        sent_lengths: torch.LongTensor,
        states: torch.Tensor,
        word_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        device = start_ids.device
        batch_size, max_len = states.size()[:2]
        num_spans = start_ids.size(1)
        max_span_width = self._hparams.max_span_width
        batch_offset = torch.arange(batch_size, device=device) * max_len
        span_indices = torch.arange(max_span_width, device=device)
        # span_indices: (batch_size, num_spans, max_span_width)
        span_indices = (
            span_indices.expand(batch_size, num_spans, -1)
            + start_ids.unsqueeze(-1)
            + batch_offset.view(-1, 1, 1)
        )
        # valid_spans: (batch_size, num_spans)
        valid_spans = end_ids < sent_lengths.unsqueeze(-1)
        # valid_spans_idx: (total_spans)
        valid_spans_idx = valid_spans.view(-1).nonzero().view(-1)
        # flat_span_indices: (total_spans, max_span_width)
        flat_span_indices = torch.index_select(
            span_indices.view(-1, max_span_width), dim=0, index=valid_spans_idx
        )

        # flat_sent_lengths: (total_spans)
        flat_sent_lengths = torch.index_select(
            (
                torch.min(end_ids + 1, sent_lengths.unsqueeze(-1))
                + batch_offset.unsqueeze(-1)
            ).view(-1),
            dim=0,
            index=valid_spans_idx,
        )
        # flat_mask: (total_spans, max_span_width)
        flat_mask = flat_span_indices < flat_sent_lengths.unsqueeze(-1)
        flat_span_indices *= flat_mask.type_as(flat_span_indices)

        # span_word_inputs: (total_spans, max_span_width, word_input_dim)
        span_word_inputs = torch.index_select(
            word_inputs.view(-1, word_inputs.size(-1)),
            dim=0,
            index=flat_span_indices.view(-1),
        ).view(*flat_span_indices.size(), -1)

        # logits: (batch_size, max_len)
        logits = self.head_attention(states).squeeze(-1)
        # flat_span_logits: (total_spans, max_span_width)
        flat_span_logits = torch.index_select(
            logits.view(-1), dim=0, index=flat_span_indices.view(-1)
        ).view(flat_span_indices.size())
        masked_span_logits = flat_span_logits - 1e10 * (~flat_mask).type_as(
            flat_span_logits
        )
        weights = torch.softmax(masked_span_logits, dim=-1)

        # weighted_inputs: (total_spans, max_span_width, word_input_dim)
        weighted_inputs = span_word_inputs * weights.unsqueeze(-1)
        # soft_head: (total_spans, word_input_dim)
        soft_head = torch.sum(weighted_inputs, dim=1)
        # indices: (batch_size, num_spans)
        indices = torch.cumsum(valid_spans.view(-1).type(torch.long), dim=0) - 1
        indices = torch.clamp_min(indices, 0).view_as(valid_spans)

        return soft_head, indices

    class ReturnType(TypedDict):
        loss: torch.Tensor
        total_scores: torch.Tensor
        start_ids: torch.LongTensor
        end_ids: torch.LongTensor
        predicates: torch.LongTensor

    def _arange(self, *args, **kwargs):
        return torch.arange(*args, device=self._device, **kwargs)

    def forward(self, inputs: tx.data.Batch) -> ReturnType:
        # Compute embeddings and recurrent states.
        char_embed = self.char_cnn(inputs.text)
        with torch.no_grad():
            # A workaround for freezing embeddings.
            word_embed = self.word_embed(inputs.text_ids)
            head_embed = self.head_embed(inputs.text_ids)
        context_embeds = [word_embed, char_embed]
        head_embeds = [head_embed, char_embed]
        if self.elmo is not None:
            char_ids = self._elmo_char_ids_fn(inputs.text).to(self._device)
            elmo_embed = self.elmo(char_ids)["elmo_representations"][0]
            context_embeds.append(elmo_embed)
        # *word_inputs: (batch_size, max_len, word_input_dim)
        lstm_word_inputs = self.embed_dropout(torch.cat(context_embeds, dim=-1))
        word_inputs = self.embed_dropout(torch.cat(head_embeds, dim=-1))

        # states: (batch_size, max_len, hidden_dim)
        states = self.lstm(lstm_word_inputs, inputs.length)

        # Create span indices.
        batch_size, max_len = inputs.text_ids.size()
        max_span = self._hparams.max_span_width
        # *_ids: (batch_size, max_num_spans)
        # max_num_spans ~= max_len * max_span
        start_ids, end_ids, span_length = self._create_span_indices(
            batch_size, max_len, max_span
        )
        # Create soft head representation weights.
        # head_attn_cache, head_attn_index = self._compute_soft_head_attention(
        (
            head_attn_cache,
            head_attn_index,
        ) = self._compute_soft_head_attention_brute(
            start_ids, end_ids, inputs.length, states, word_inputs
        )

        # Compute argument & predicate scores.
        span_length_embed = self.embed_dropout(self.span_length_embed.embedding)
        cache_inputs = [states, states, head_attn_cache, span_length_embed]
        pred_indices = self._arange(max_len).expand(batch_size, -1)
        with self.argument_mlp.cache_results(
            cache_inputs
        ), self.predicate_mlp.cache_results([states]):
            # arg_scores: (batch_size, max_num_spans)
            arg_scores = self.argument_mlp(
                [start_ids, end_ids, head_attn_index, span_length]
            ).squeeze(-1)
            # pred_scores: (batch_size, max_len)
            pred_scores = self.predicate_mlp([pred_indices]).squeeze(-1)

        # Beam pruning of arguments & predicates.
        # topk_*: (batch_size, max_arguments)
        max_arguments = math.ceil(self._hparams.argument_ratio * max_len)
        num_arguments = torch.ceil(
            self._hparams.argument_ratio * inputs.length.float()
        ).long()
        topk_arg_scores, topk_arg_indices = torch.topk(
            arg_scores, k=max_arguments, dim=1, sorted=True
        )
        topk_start_ids, topk_end_ids, topk_attn_index = utils.batch_gather(
            [start_ids, end_ids, head_attn_index], index=topk_arg_indices
        )
        topk_span_length = topk_end_ids - topk_start_ids

        # topk_pred_*: (batch_size, max_predicates)
        max_predicates = math.ceil(self._hparams.predicate_ratio * max_len)
        num_predicates = torch.ceil(
            self._hparams.predicate_ratio * inputs.length.float()
        ).long()
        topk_pred_scores, topk_pred_indices = torch.topk(
            pred_scores, k=max_predicates, dim=1, sorted=True
        )

        # Compute label scores for pruned argument-predicate pairs.
        with self.span_label_mlp.cache_results(cache_inputs + [states]):
            # label_scores:
            #   (batch_size, max_predicates * max_arguments, num_labels)
            label_scores = self.span_label_mlp(
                [
                    topk_start_ids.repeat(1, max_predicates),
                    topk_end_ids.repeat(1, max_predicates),
                    tx.utils.map_structure(
                        lambda x: x.repeat(1, max_predicates)
                        if isinstance(x, torch.Tensor)
                        else x,
                        topk_attn_index,
                    ),
                    topk_span_length.repeat(1, max_predicates),
                    topk_pred_indices.repeat_interleave(max_arguments, dim=1),
                ]
            )

        # Compute log-probabilities.
        total_scores = (
            topk_arg_scores.repeat(1, max_predicates).unsqueeze(-1)
            + topk_pred_scores.repeat_interleave(
                max_arguments, dim=1
            ).unsqueeze(-1)
            + label_scores
        )
        total_scores = torch.cat(
            [
                total_scores.new_zeros(*total_scores.size()[:-1], 1),
                total_scores,
            ],
            dim=-1,
        )
        gold_labels = self._filter_labels(
            topk_start_ids, topk_end_ids, topk_pred_indices, inputs.srl
        )

        # Compute masked loss.
        # unmasked_loss: (batch_size, max_predicates * max_arguments)
        unmasked_loss = F.cross_entropy(
            total_scores.view(-1, total_scores.size(-1)),
            gold_labels.view(-1),
            reduction="none",
        )
        # pred_*_mask: (batch_size, max_predicates)
        pred_index_mask = topk_pred_indices < inputs.length.unsqueeze(-1)
        pred_topk_mask = self._arange(max_predicates).unsqueeze(
            0
        ) < num_predicates.unsqueeze(1)
        # arg_*_mask: (batch_size, max_arguments)
        arg_index_mask = topk_end_ids < inputs.length.unsqueeze(-1)
        arg_topk_mask = self._arange(max_arguments).unsqueeze(
            0
        ) < num_arguments.unsqueeze(1)
        loss_mask = (
            (arg_index_mask & arg_topk_mask).unsqueeze(1)
            & (pred_index_mask & pred_topk_mask).unsqueeze(2)
        ).view(-1)
        loss = torch.sum(unmasked_loss * loss_mask.type_as(unmasked_loss))
        # loss = loss / batch_size
        return {
            "loss": loss,
            "total_scores": total_scores,
            "start_ids": topk_start_ids,
            "end_ids": topk_end_ids,
            "predicates": topk_pred_indices,
        }

    _CORE_ARGS = {
        f"{prefix}{arg}": 1 << idx
        for prefix in ["A", "ARG"]
        for idx, arg in enumerate("012345A")
    }

    def _dp_decode(
        self,
        max_len: int,
        pred_idx: int,
        start_ids: List[int],
        end_ids: List[int],
        argmax_labels: List[int],
        label_scores: List[float],
        enforce_constraint: bool = False,
    ) -> List[Span]:
        # Map positions to list of span indices for quick lookup during DP.
        spans_ending_at: Dict[int, List[int]] = defaultdict(list)
        for idx in range(  # pylint: disable=consider-using-enumerate
            len(end_ids)
        ):
            if argmax_labels[idx] == 0:  # ignore null spans
                continue
            if start_ids[idx] <= pred_idx <= end_ids[idx]:
                # Skip spans overlapping with the predicate.
                continue
            if end_ids[idx] >= max_len:
                # Skip invalid spans longer than the sentence.
                continue
            spans_ending_at[end_ids[idx]].append(idx)
        if all(len(spans) == 0 for spans in spans_ending_at.values()):
            return []  # no spans at all, just return
        if enforce_constraint:
            label_states = [
                self._CORE_ARGS.get(self.label_inverse_vocab[label], -1)
                if label != 0
                else -1
                for label in argmax_labels
            ]
        else:
            # ignore constraints
            label_states = [-1] * len(argmax_labels)

        # Perform DP.
        # Each state is a tuple (time, core_args), where `core_args` is the set
        # of core arguments (ARGA, ARG0 to ARG5) previously selected,
        # represented in binary (so {ARG0, ARG2, ARG5} would be
        # 2^0 + 2^2 + 2^5 = 37).
        max_scores = [{0: 0.0}]
        # only record selected spans
        best_span_indices: List[Dict[int, int]] = [{}]

        for idx in range(max_len):
            cur_scores = max_scores[-1].copy()
            cur_span_idx = {}
            for span_idx in spans_ending_at[idx]:
                label_state = label_states[span_idx]
                prev_states = max_scores[start_ids[span_idx]]
                for state, prev_score in prev_states.items():
                    if label_state != -1 and (label_state & state != 0):
                        # A core argument of this type has already been selected
                        continue
                    score = prev_score + label_scores[span_idx]
                    new_state = state | label_state
                    if score > cur_scores.get(new_state, 0):
                        cur_scores[new_state] = score
                        cur_span_idx[new_state] = span_idx
            max_scores.append(cur_scores)
            best_span_indices.append(cur_span_idx)

        # Backtrack to obtain optimal span choices.
        srl = []
        pos = max_len
        state = max(
            (score, state) for state, score in max_scores[max_len].items()
        )[1]
        while pos > 0:
            best_span_idx = best_span_indices[pos].get(state, None)
            if best_span_idx is not None:
                assert end_ids[best_span_idx] == pos - 1
                srl.append(
                    Span(
                        start_ids[best_span_idx],
                        end_ids[best_span_idx],
                        self.label_inverse_vocab[argmax_labels[best_span_idx]],
                    )
                )
                pos = start_ids[best_span_idx]
                if label_states[best_span_idx] != -1:
                    state &= ~label_states[best_span_idx]
            else:
                pos -= 1
        return srl

    @torch.no_grad()
    def decode(
        self, inputs: tx.data.Batch, enforce_constraint: bool = False
    ) -> List[Dict[int, List[Span]]]:
        r"""Performs optimal decoding with dynamic programming.

        :returns: A nested structure of SRL spans, representing the (inner) list
            of spans for each predicate (middle `dict`) and for each example in
            the batch (outer list).
        """
        result_dict = self.forward(inputs)
        start_ids = result_dict["start_ids"].cpu().numpy()
        end_ids = result_dict["end_ids"].cpu().numpy()
        predicates = result_dict["predicates"].cpu().numpy()
        batch_size, num_arguments = start_ids.shape
        num_predicates = predicates.shape[1]

        total_scores = result_dict["total_scores"].view(
            batch_size, num_predicates, num_arguments, -1
        )
        label_scores, argmax_label = torch.max(total_scores, dim=3)
        argmax_label = argmax_label.cpu().numpy()
        label_scores = label_scores.cpu().numpy()
        sent_lengths = inputs.length.cpu().numpy()

        # Do DP one example at a time...
        batch_srl = []
        for b_idx in range(batch_size):
            cur_srl: Dict[int, List[Span]] = {}
            # ... and one predicate at a time.
            for pred_idx, predicate in enumerate(predicates[b_idx]):
                if predicate >= inputs.length[b_idx]:
                    # Skip invalid predicates outside the sentence.
                    continue
                srl = self._dp_decode(
                    sent_lengths[b_idx],
                    predicate,
                    start_ids[b_idx],
                    end_ids[b_idx],
                    argmax_label[b_idx, pred_idx],
                    label_scores[b_idx, pred_idx],
                    enforce_constraint,
                )
                if len(srl) > 0:
                    cur_srl[predicate] = srl
            batch_srl.append(cur_srl)
        return batch_srl
