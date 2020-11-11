#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:34:02 2017

@author: elena
"""
from __future__ import print_function

import codecs
# from nltk import word_tokenize
import collections
import operator
import os
import re
import shutil

import h5py
import numpy as np
import tensorflow as tf


# import Dataset as dset
# import nltk

def variable_summaries(var):
    '''
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    From https://www.tensorflow.org/get_started/summaries_and_tensorboard
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def load_parameters_from_file(parameters_file_name):
    d = {}
    with open(parameters_file_name) as f:
        for line in f:
            #  if not line.strip():
            (key, val) = line.split()
            if is_number(val):
                d[key] = int(val)
                continue
            if is_boolean(val):
                d[key] = bool(val)
                continue
            d[key] = val
    return d


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_boolean(s):
    return bool(s in ['True', 'False'])


def get_features_for_sentence(dataset_adress, sentence_number):
    reading_table = h5py.File(dataset_adress, 'r')
    word_features = reading_table["word-features"]
    sentences_words = reading_table["sentences-words"]
    current_sentence = sentences_words[sentence_number]
    indicies = list(current_sentence)
    extracted_feature_matrix = word_features[indicies[0]:indicies[-1] + 1, :]

    list_features = extracted_feature_matrix.tolist()
    # list_features=[x.tolist() for x in list_features]
    return list_features


# z=get_features_for_sentence(
# "./data-fordataset/CATEGORIES-INCLUDED/train.hdf5",0)
# print (len(z))
# print (len(z[0]))


def get_size_of_features(main_data_file_address):
    f = codecs.open(main_data_file_address, 'r', 'UTF-8')
    size_of_the_features_vector = 0
    for line in f:
        line = line.strip().split(' ')
        num_of_elem_line = len(line)
        token_features = [x for ind, x in enumerate(line) if
                          ind not in [num_of_elem_line - 1,
                                      num_of_elem_line - 2] and x != ""]
        size_of_the_features_vector = len(token_features)
        # print (size_of_the_features_vector)
        break
    return size_of_the_features_vector


def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def copytree(src, dst, symlinks=False, ignore=None):
    '''
    http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire
    -directory-of-files-into-an-existing-directory-using-pyth
    '''
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def string_to_list_of_lists(
        string):  # NOT IN USE, was used for old feature represntation
    list_of_tokens = []
    feature_list = []
    features = string.split(" ")
    for feature in features:
        if feature == "#newtoken#":
            list_of_tokens.append(feature_list)
            feature_list = []
        else:
            try:
                feature_list.append(float(feature))
            except ValueError:
                continue
    return list_of_tokens


def get_valid_dataset_filepaths(parameters):
    dataset_filepaths = {}
    for dataset_type in ['train', 'test']:
        dataset_filepaths[dataset_type] = os.path.join(
            parameters['dataset_text_folder'],
            '{0}.txt'.format(dataset_type))
    return dataset_filepaths


def remove_file_name_from_the_path_string(path_string):
    get_separator = os.sep
    break_path = path_string.split(get_separator)
    new_path = [n for idx, n in enumerate(break_path) if
                idx != len(break_path) - 1]
    new_path = get_separator.join(new_path)
    return new_path


def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode == 'key':
        return collections.OrderedDict(sorted(dictionary.items(),
            key=operator.itemgetter(0),
            reverse=reverse))
    elif mode == 'value':
        return collections.OrderedDict(sorted(dictionary.items(),
            key=operator.itemgetter(1),
            reverse=reverse))
    elif mode == 'key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
            reverse=reverse))
    elif mode == 'value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
            key=lambda x: (x[1], x[0]),
            reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")


def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an
    -ordereddict-using-its-constructor-such-that-it-retain
    '''
    # print('type(dictionary): {0}'.format(type(dictionary)))
    # pylint: disable=unidiomatic-typecheck
    if type(dictionary) is collections.OrderedDict:
        # print(type(dictionary))
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}


def is_token_in_pretrained_embeddings(token, all_pretrained_tokens):
    # return token in all_pretrained_tokens or \
    # pylint: disable=anomalous-backslash-in-string
    return re.sub('\\d', '0', token.lower()) in all_pretrained_tokens


def remove_bio_from_label_name(label_name):
    if label_name[:2] in ['B-', 'I-', 'E-', 'S-']:
        # print (label_name[:2])
        new_label_name = label_name[2:]
    else:
        assert label_name == 'O'
        new_label_name = label_name
    return new_label_name


def load_pretrained_token_embeddings(parameters):
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'],
        'r', 'UTF-8')
    count = -1
    token_to_vector = {}
    for cur_line in file_input:
        count += 1
        # if count > 1000:break
        cur_line = cur_line.strip()
        if not len(cur_line):
            continue
        token = cur_line[0]
        vector = np.array([float(x) for x in cur_line[1:]])
        token_to_vector[token] = vector
    file_input.close()
    return token_to_vector  # Dictionary of token-vectors


def load_tokens_from_pretrained_token_embeddings(parameters):
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'],
        'r', 'UTF-8')
    count = -1
    tokens = set()
    number_of_loaded_word_vectors = 0
    for cur_line in file_input:
        count += 1
        cur_line = cur_line.split(' ')
        if len(cur_line) == 0:
            continue
        token = cur_line[0]
        tokens.add(token)
        number_of_loaded_word_vectors += 1
    file_input.close()
    return tokens


def pad_list(old_list, padding_size,
        padding_value):  # ONE SIDED, might have issues for BIDIRECTIONAL
    # LSTM BATCH NORMALIZATION
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in
    -python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list)
    return old_list + [padding_value] * (padding_size - len(old_list))


def get_parsed_conll_output(conll_output_filepath):
    conll_output = [
        l.rstrip().replace('%', '').replace(';', '').replace(':', '').strip()
        for l in
        codecs.open(conll_output_filepath, 'r', 'utf8')]
    parsed_output = {}
    line = conll_output[1].split()
    parsed_output['all'] = {'accuracy': float(line[1]),
                            'precision': float(line[3]),
                            'recall': float(line[5]),
                            'f1': float(line[7])}
    total_support = 0
    for line in conll_output[2:]:
        line = line.split()
        phi_type = line[0].replace('_', '-')
        # print (phi_type)
        # print (line)
        support = int(line[7])
        total_support += support
        parsed_output[phi_type] = {'precision': float(line[2]),
                                   'recall': float(line[4]),
                                   'f1': float(line[6]),
                                   'support': support}
    parsed_output['all']['support'] = total_support

    print(parsed_output['all'])
    return parsed_output

# z=get_parsed_conll_output(
# "./RESULTS/CONLL-TEST/epoche_1.txt_conll_evaluation.txt")


# extract_from_the_tree("FIXED_I2B2_XML/i2b2_2012/training/28.xml","")

# tokenize=word_tokenize(test)
# print tokenize
# write_all_files_into_one_file("FIXED_I2B2_XML/i2b2_2012/training/") # Add
# flag "Deal with double qotes  as if they were marked -1,1 text global
# span-move+2
# timeexp,spanlist = extract_from_the_tree("28.xml","28.xml.txt")
# z=map_time_exp_to_text(spanlist,timeexp)
# write_to_file_pseudo_conLL(z)

# opening_path={'token_pretrained_embedding_filepath':'glove.6B.100d.txt',
# "freeze_token_embeddings" :'True'}
# tokens=load_tokens_from_pretrained_token_embeddings(opening_path)
# horrible_list=load_pretrained_token_embeddings(opening_path)

# print horrible_list["cancer"]
# print tokens
