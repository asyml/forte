"""
Utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import List

import texar as tx

from examples.content_rewriter.model.data2text.data_utils import \
    extract_entities

# load all entities

e2e_ents = set()

get_scope_name_of_train_op = 'train_{}'.format
get_scope_name_of_summary_op = 'summary_{}'.format

ref_strs = ['', '_ref']
sent_fields = ['y_aux', 'y_ref']

x_fields: List[str] = ['value', 'type', 'associated']
x_strs = ['x', 'x_ref']
y_strs = ['y_aux', 'y_ref']
y_tgt_strs = ['y_ref']


def load_e2e_ents(e2e_vocab_path: str):
    with open(e2e_vocab_path, 'r') as f:
        all_vocb = f.readlines()
        for vocab in all_vocb:
            e2e_ents.add(vocab.strip('\n'))


class DataItem(collections.namedtuple('DataItem', x_fields)):  # type: ignore
    def __str__(self):
        return '|'.join(map(str, self))


def pack_sd(paired_texts):
    return [DataItem(*_) for _ in zip(*paired_texts)]


def batchize(func):
    def batchized_func(*inputs):
        return [func(*paired_inputs) for paired_inputs in zip(*inputs)]

    return batchized_func


def strip_special_tokens_of_list(text):
    return tx.utils.strip_special_tokens(text, is_token_list=True)


batch_strip_special_tokens_of_list = batchize(strip_special_tokens_of_list)


def replace_data_in_sent(sent, token="<UNK>"):
    data_type = 'e2e'
    if data_type == 'e2e':
        datas = extract_entities(sent, e2e_ents)
        datas.sort(key=lambda data: data.start, reverse=True)
        for data in datas:
            sent[data.start] = token
    return sent


def corpus_bleu(list_of_references, hypotheses, **kwargs):
    list_of_references = [
        list(map(replace_data_in_sent, refs))
        for refs in list_of_references]
    hypotheses = list(map(replace_data_in_sent, hypotheses))
    return tx.evals.corpus_bleu_moses(
        list_of_references, hypotheses,
        lowercase=True, return_all=False,
        **kwargs)


def read_sents_from_file(file_name):
    with open(file_name, 'r') as f:
        return list(map(str.split, f))


def divide_or_const(a, b, c=0.):
    try:
        return a / b
    except ZeroDivisionError:
        return c
