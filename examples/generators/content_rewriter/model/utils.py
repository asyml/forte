"""
Utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import texar as tx
from tensorflow.contrib.seq2seq import tile_batch
from data2text.data_utils import get_train_ents, extract_entities, extract_numbers


# load all entities
all_ents, players, teams, cities = get_train_ents(path=os.path.join("data2text", "rotowire"), connect_multiwords=True)
# all_ents = set()
# with open('e2e_data/x_value.vocab.txt', 'r') as f:
#     all_vocb = f.readlines()
#     for vocab in all_vocb:
#         all_ents.add(vocab.strip('\n'))

get_scope_name_of_train_op = 'train_{}'.format
get_scope_name_of_summary_op = 'summary_{}'.format


x_fields = ['value', 'type', 'associated']
x_strs = ['x', 'x_ref']
y_strs = ['y_aux', 'y_ref']
ref_strs = ['', '_ref']

class DataItem(collections.namedtuple('DataItem', x_fields)):
    def __str__(self):
        return '|'.join(map(str, self))

def pack_x(paired_texts):
    return [DataItem(*_) for _ in zip(*paired_texts)]

def batchize(func):
    def batchized_func(*inputs):
        return [func(*paired_inputs) for paired_inputs in zip(*inputs)]
    return batchized_func

def strip_special_tokens_of_list(text):
    return tx.utils.strip_special_tokens(text, is_token_list=True)

batch_strip_special_tokens_of_list = batchize(strip_special_tokens_of_list)

def replace_data_in_sent(sent, token="<UNK>"):
    datas = extract_entities(sent, all_ents) + extract_numbers(sent)
    datas.sort(key=lambda data: data.start, reverse=True)
    for data in datas:
        sent[data.start: data.end] = token
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

def read_x(data_prefix, ref_flag, stage):
    ref_str = ref_strs[ref_flag]
    return list(map(
        lambda paired_sents: list(map(
            lambda tup: DataItem(*tup),
            zip(*paired_sents))),
        zip(*map(
            lambda field: read_sents_from_file(
                '{}{}{}.{}.txt'.format(data_prefix, field, ref_str, stage)),
            sd_fields))))

def read_y(data_prefix, ref_flag, stage):
    ref_str = ref_strs[ref_flag]
    field = sent_fields[0]
    return read_sents_from_file(
        '{}{}{}.{}.txt'.format(data_prefix, field, ref_str, stage))

def divide_or_const(a, b, c=0.):
    try:
        return a / b
    except ZeroDivisionError:
        return c
