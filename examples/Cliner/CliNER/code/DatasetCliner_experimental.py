#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:55:40 2017

@author: elena
"""

import collections
# import token
import os
import pickle
import random
# import utils_nlp
import time

import numpy as np
import sklearn.preprocessing

from examples.Cliner.CliNER.code import helper_dataset


def lists_to_dataset_structure(sentences_tokens, sentence_tags,
        total_token_counter, token_count, label_count,
        character_count):
    labels = []
    tokens = []
    new_label_sequence = []
    new_token_sequence = []

    features = ""
    feature_file_name = ""
    feature_vector_size = 0

    for idx, sentence in enumerate(sentences_tokens):
        for token_idx, token_i in enumerate(sentence):
            new_token_sequence.append(token_i)
            new_label_sequence.append(sentence_tags[idx][token_idx])

            token_count[token_i] += 1
            label_count[sentence_tags[idx][token_idx]] += 1

            if token_idx == len(sentence) - 1:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
                new_token_sequence = []
                new_label_sequence = []
            # FEATURES ARE NOT SUPPORTED:  Can be done if we are getting a \
            # third list that looks like [[f1,f2,f3],[f1,f2,f3]... for each \
            # token]
            token_features = []
            features_as_array = np.array(token_features,
                dtype=np.dtype('int32'))
            features_as_array = features_as_array.reshape(
                (features_as_array.shape[0], 1))
            features_as_array = np.transpose(features_as_array)

            features = ""
            feature_file_name = ""
            feature_vector_size = 0

            total_token_counter += 1
            for character in token_i:
                character_count[character] += 1

    return labels, tokens, token_count, label_count, character_count, \
           features, feature_file_name, feature_vector_size


class Dataset():
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def _parse_dataset(self, sentences_list, tags_list, Not_here=False):

        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        total_token_counter = 0

        tokens = []
        labels = []
        features = []

        if not Not_here:
            labels, tokens, token_count, label_count, character_count, \
            features, feature_file_name, feature_vector_size = \
                lists_to_dataset_structure(
                    sentences_list, tags_list, total_token_counter,
                    token_count,
                    label_count, character_count)

        return labels, tokens, token_count, label_count, character_count, \
               features, feature_file_name, feature_vector_size

    def _convert_to_indices(self, dataset_types):
        # Frank and Jennies Function

        tokens = self.tokens
        labels = self.labels
        token_to_index = self.token_to_index
        character_to_index = self.character_to_index
        label_to_index = self.label_to_index
        index_to_label = self.index_to_label

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        characters = {}
        token_lengths = {}
        character_indices = {}
        character_indices_padded = {}
        for dataset_type in dataset_types:
            print(dataset_type)
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            character_indices_padded[dataset_type] = []

            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([
                    token_to_index.get(token, self.UNK_TOKEN_INDEX)
                    for token in token_sequence
                ])
                characters[dataset_type].append(
                    [list(token) for token in token_sequence])
                character_indices[dataset_type].append([[
                    character_to_index.get(
                        character,
                        random.randint(1, max(self.index_to_character.keys())))
                    for character in token
                ] for token in token_sequence])
                token_lengths[dataset_type].append(
                    [len(token) for token in token_sequence])
                longest_token_length_in_sequence = \
                    max(token_lengths[dataset_type][-1])
                character_indices_padded[dataset_type].append([
                    helper_dataset.pad_list(temp_token_indices,
                        longest_token_length_in_sequence,
                        self.PADDING_CHARACTER_INDEX) for
                    temp_token_indices in character_indices[dataset_type][-1]
                ])

            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append(
                    [label_to_index[label] for label in label_sequence])

        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(index_to_label.keys()) + 1))
        label_vector_indices = {}
        for dataset_type in dataset_types:
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(
                    label_binarizer.transform(label_indices_sequence))

        return token_indices, label_indices, character_indices_padded, \
               character_indices, token_lengths, characters, \
               label_vector_indices

    def update_dataset(self, dataset_types, Datasets_tokens, Datasets_labels):
        """
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', \
        'deploy'
        Overwrites the data of type specified in dataset_types using the \
        existing token_to_index, character_to_index, and label_to_index
        mappings.
        """

        # def _parse_dataset(self, dataset_filepath, dataset_type, \
        # sentences_list=[],tags_list=[], Not_here=False):
        for dataset_type in dataset_types:
            print(dataset_type)
            self.labels[dataset_type], self.tokens[dataset_type], _, _, \
            _, _, _, _ = self._parse_dataset(
                Datasets_tokens[
                    dataset_type],
                Datasets_labels[
                    dataset_type])

        token_indices, label_indices, character_indices_padded, \
        character_indices, token_lengths, characters, label_vector_indices \
            = self._convert_to_indices(
            dataset_types)

        self.token_indices.update(token_indices)
        self.label_indices.update(label_indices)
        self.character_indices_padded.update(character_indices_padded)
        self.character_indices.update(character_indices)
        self.token_lengths.update(token_lengths)
        self.characters.update(characters)
        self.label_vector_indices.update(label_vector_indices)

    def load_dataset(self,
            avaliable_datasets_sent,
            avaliable_datasets_labels,
            parameters,
            token_to_vector=None,
            pretrained_dataset=None):
        """
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test',
        'deploy'
        """
        start_time = time.time()
        print('Load dataset... \n')
        if parameters['token_pretrained_embedding_filepath'] != '':
            if not token_to_vector:
                token_to_vector = \
                    helper_dataset.load_pretrained_token_embeddings(
                    parameters)
        else:
            token_to_vector = {}

        all_tokens_in_pretraining_dataset = []
        all_characters_in_pretraining_dataset = []

        if parameters['use_pretrained_model']:

            if not pretrained_dataset:
                temp_pretrained_dataset_adress = \
                    parameters['model_folder'] + os.sep + "dataset.pickle"
                pretraining_dataset = \
                    pickle.load(open(temp_pretrained_dataset_adress, "rb"))
                print("Pre-loading Pre-trained dataset objects")
            else:
                pretraining_dataset = pretrained_dataset
                print("Pretrained dataset was pre-loaded")

            all_tokens_in_pretraining_dataset = \
                pretraining_dataset.index_to_token.values()
            all_characters_in_pretraining_dataset = \
                pretraining_dataset.index_to_character.values()

        remap_to_unk_count_threshold = 1
        # pylint: disable=attribute-defined-outside-init
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = 'UNK'
        self.unique_labels = []
        labels = {}
        tokens = {}
        label_count = {}
        token_count = {}
        character_count = {}

        features = {}
        features_file_names = {}
        feature_vector_size = {}
        # deploy

        for dataset_type in ['train', 'valid', 'test', 'deploy']:
            Not_here = False

            if dataset_type not in avaliable_datasets_sent:
                Not_here = True
            if not Not_here:
                labels[dataset_type], tokens[dataset_type], \
                token_count[dataset_type], label_count[dataset_type], \
                character_count[dataset_type], features[dataset_type], \
                features_file_names[dataset_type], \
                feature_vector_size[dataset_type] \
                    = self._parse_dataset(
                    sentences_list=avaliable_datasets_sent
                    [dataset_type],
                    tags_list=avaliable_datasets_labels[
                        dataset_type])

            if Not_here:
                labels[dataset_type], tokens[dataset_type], \
                token_count[dataset_type], label_count[dataset_type], \
                character_count[dataset_type], features[dataset_type], \
                features_file_names[dataset_type], \
                feature_vector_size[dataset_type] \
                    = self._parse_dataset(sentences_list=[], tags_list=[])

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + \
                     list(token_count['valid'].keys()) + list(
            token_count['test'].keys()) + \
                     list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'][token] + \
                                        token_count['valid'][token] + \
                                        token_count['test'][token] \
                                        + token_count['deploy'][token]

        if parameters['load_all_pretrained_token_embeddings']:
            for token in token_to_vector:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1
            for token in all_tokens_in_pretraining_dataset:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + \
                         list(character_count['valid'].keys()) + \
                         list(character_count['test'].keys()) + \
                         list(character_count['deploy'].keys()):
            character_count['all'][character] = \
                character_count['train'][character] + character_count['valid'][
                    character] + character_count['test'][character] + \
                character_count['deploy'][character]

        for character in all_characters_in_pretraining_dataset:
            if character not in character_count['all']:
                character_count['all'][character] = -1
                character_count['train'][character] = -1

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + \
                         list(label_count['valid'].keys()) + list(
            label_count['test'].keys()) + \
                         list(label_count['deploy'].keys()):
            label_count['all'][character] = \
                label_count['train'][character] + \
                label_count['valid'][character] + \
                label_count['test'][character] + label_count['deploy'][
                    character]

        token_count['all'] = helper_dataset.order_dictionary(token_count['all'],
            'value_key',
            reverse=True)
        label_count['all'] = helper_dataset.order_dictionary(label_count['all'],
            'key',
            reverse=False)
        character_count['all'] = helper_dataset.order_dictionary(
            character_count['all'],
            'value',
            reverse=True)
        if self.verbose:
            print('character_count[\'all\']: {0}'.format(
                character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        if self.verbose:
            print("parameters['remap_unknown_tokens_to_unk']: {0}".format(
                parameters['remap_unknown_tokens_to_unk']))
        if self.verbose:
            print("len(token_count['train'].keys()): {0}".format(
                len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX:
                iteration_number += 1

            # pylint: disable=too-many-function-args
            if parameters['remap_unknown_tokens_to_unk'] == 1 and \
                    (token_count['train'][token] == 0 or
                     parameters['load_only_pretrained_token_embeddings']) and \
                    not helper_dataset.is_token_in_pretrained_embeddings(token,
                        token_to_vector,
                        parameters) and \
                    token not in all_tokens_in_pretraining_dataset:
                token_to_index[token] = self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])

        labels_without_bio = set()
        for label in label_count['all'].keys():
            new_label = helper_dataset.remove_bio_from_label_name(label)
            labels_without_bio.add(new_label)
        for label in labels_without_bio:
            if label == 'O':
                continue
            if parameters['tagging_format'] == 'bioes':
                prefixes = ['B-', 'I-', 'E-', 'S-']
            else:
                prefixes = ['B-', 'I-']
            for prefix in prefixes:
                l = prefix + label
                if l not in label_count['all']:
                    label_count['all'][l] = 0
        label_count['all'] = helper_dataset.order_dictionary(label_count['all'],
            'key',
            reverse=False)

        if parameters['use_pretrained_model']:

            print("USE_PRETRAINED_MODEL ACTIVE")
            # pylint: disable=attribute-defined-outside-init
            self.unique_labels = sorted(list(
                pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError(
                        "The label {0} does not exist in"
                        "the pretraining dataset. ".format(label) +
                        "Please ensure that only the"
                        "following labels exist in the dataset: {0}".format(
                            ', '.join(self.unique_labels)))
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX:
                iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        token_to_index = helper_dataset.order_dictionary(token_to_index,
            'value',
            reverse=False)
        if self.verbose:
            print('token_to_index: {0}'.format(token_to_index))
        index_to_token = helper_dataset.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1:
            index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        if self.verbose:
            print('index_to_token: {0}'.format(index_to_token))

        label_to_index = helper_dataset.order_dictionary(label_to_index,
            'value',
            reverse=False)
        index_to_label = helper_dataset.reverse_dictionary(label_to_index)

        character_to_index = helper_dataset.order_dictionary(character_to_index,
            'value',
            reverse=False)
        index_to_character = helper_dataset.reverse_dictionary(
            character_to_index)
        # pylint: disable=attribute-defined-outside-init
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        self.tokens = tokens
        self.labels = labels

        dataset_types = ['train', 'test', 'valid', 'deploy']
        token_indices, label_indices, character_indices_padded, \
        character_indices, token_lengths, characters, label_vector_indices \
            = self._convert_to_indices(
            dataset_types)

        self.token_indices = token_indices
        self.label_indices = label_indices
        self.character_indices_padded = character_indices_padded
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.characters = characters
        self.label_vector_indices = label_vector_indices

        self.number_of_classes = max(self.index_to_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.alphabet_size = max(self.index_to_character.keys()) + 1

        # unique_labels_of_interest is used to compute F1-scores.
        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')

        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])

        self.infrequent_token_indices = infrequent_token_indices

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

        self.feature_vector_size = 0

        return token_to_vector
