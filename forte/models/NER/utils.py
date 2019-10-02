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
