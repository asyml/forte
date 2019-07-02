from typing import Tuple, Dict
import importlib
import logging
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import texar
from nlp.pipeline.models.NER.vocabulary_processor import Alphabet
from texar.modules.embedders import WordEmbedder
from conditional_random_field import ConditionalRandomField
config_model = importlib.import_module("config_model")


class BiRecurrentConvCRF(nn.Module):
    def __init__(self,
                 word_alphabet: Alphabet,
                 char_alphabet: Alphabet,
                 tag_alphabet: Alphabet,
                 embedding_dict: Dict,
                 embedding_dim: int):
        super().__init__()

        for word in embedding_dict:
            if word not in word_alphabet.instance2index:
                word_alphabet.add(word)

        def construct_word_embedding_table():
            scale = np.sqrt(3.0 / embedding_dim)
            table = np.empty(
                [word_alphabet.size(), embedding_dim], dtype=np.float32
            )
            oov = 0
            for word, index in word_alphabet.items():
                if word in embedding_dict:
                    embedding = embedding_dict[word]
                elif word.lower() in embedding_dict:
                    embedding = embedding_dict[word.lower()]
                else:
                    embedding = np.random.uniform(
                        -scale, scale, [1, embedding_dim]
                    ).astype(np.float32)
                    oov += 1
                table[index, :] = embedding
            print("oov: %d when creating word embedding" % oov)
            return torch.from_numpy(table)

        word_table = construct_word_embedding_table()
        print(f"The size of Vocabulary:{word_alphabet.size()}")

        self.word_embedder = WordEmbedder(
            vocab_size=word_alphabet.size(),
            init_value=word_table,
            hparams=config_model.word_emb,
        )

        self.char_embedder = WordEmbedder(
            vocab_size=char_alphabet.size(), hparams=config_model.char_emb
        )

        self.char_cnn = torch.nn.Conv1d(**config_model.char_cnn_conv)

        self.dropout_in = nn.Dropout2d(config_model.dropout_rate)
        # standard dropout
        self.dropout_rnn_in = nn.Dropout(config_model.dropout_rate)
        self.dropout_out = nn.Dropout(config_model.dropout_rate)

        # self.rnn = BidirectionalRNNEncoder(
        #     input_size=embedding_dim + config_model.char_cnn_conv[
        #         "out_channels"],
        #     hparams=config_model.bilstm_sentence_encoder,
        # )

        self.rnn = nn.LSTM(
            embedding_dim + config_model.char_cnn_conv["out_channels"],
            config_model.rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dense = nn.Linear(
            config_model.rnn_hidden_size * 2, config_model.output_hidden_size
        )

        self.tag_projection_layer = nn.Linear(
            config_model.output_hidden_size, tag_alphabet.size()
        )

        self.crf = ConditionalRandomField(
            tag_alphabet.size(),
            constraints=None,
            include_start_end_transitions=True,
        )
        self.initializer = config_model.initializer
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
        :param input_word:
        :param input_char:
        :param target:
        :param mask:
        :param hx:
        :return:
            Return the loss value
        """
        output, _, mask, _ = self.encode(input_word, input_char, mask, hx)

        logits = self.tag_projection_layer(output)
        log_likelihood = (
            self.crf.forward(logits, target, mask) / target.size()[0]
        )
        return -log_likelihood

    def decode(self, input_word, input_char, mask=None, hx=None):
        """

        :param input_word:
        :param input_char:
        :param mask:
        :param hx:
        :return:
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
            masks = masks[:, :lens[0]]
        else:
            masks = masks[:lens[0]]

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


def evaluate(output_file: str) -> Tuple[float, float, float, float]:
    """
    :param output_file: The file to be evaluated
    :return:
        return the metrics evaluated by the conll03_eval.v2 script
        (accuracy, precision, recall, F1)
    """
    score_file = f"{output_file}.score"
    os.system(
        "./conll03eval.v2 < %s > %s" % (output_file, score_file)
    )
    with open(score_file, "r") as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def get_logger(
    name,
    level=logging.INFO,
    formatter="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(name + ".log")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(formatter))
    logger.addHandler(fh)

    return logger


def batch_size_fn(new: Tuple, count: int, size_so_far: int):
    if count == 1:
        batch_size_fn.max_length = 0
    batch_size_fn.max_length = max(batch_size_fn.max_length, len(new[0]))
    elements = count * batch_size_fn.max_length
    return elements
