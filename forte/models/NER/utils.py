import logging
import os
import random
import re
from typing import Tuple

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(r"\d")


def normalize_digit_word(word):
    return DIGIT_RE.sub("0", word)


def load_glove_embedding(embedding_path, normalize_digits=True):
    """
    Load glove embeddings from file.

    Args:
        embedding_path:  the file to load embedding from.
        normalize_digits: whether to normalize the digits characters in token.

    Returns: embedding dict, embedding dimension, caseless

    """

    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert embedd_dim + 1 == len(
                    tokens
                ), f"glove_dim{embedd_dim} cur_dim{len(tokens)}"
            embedd = np.empty(embedd_dim, dtype=np.float32)
            embedd[:] = tokens[1:]
            word = (
                DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
            )
            embedd_dict[word] = embedd

    return embedd_dict


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
            # hack lstm
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
    Evaluate using the conll03 script.

    Args:
        output_file: The file to be evaluated

    Returns: return the metrics evaluated by the conll03_eval.v2 script
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


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
