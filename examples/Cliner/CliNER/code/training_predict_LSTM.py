#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:02:11 2017

@author: elena
"""
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from examples.Cliner.CliNER.code import helper_dataset
from examples.Cliner.CliNER.code.evaluation_LSTM import remap_labels


# import model_lstm as used_model
# import Per_token_sum as stat


def compute_train_accuracy(epoche_adress):
    "COMPUTE T_A"
    f = open(epoche_adress, 'r')
    correctly_predicted_tokens = 0
    for _, line in enumerate(f):
        if line == "\n":
            continue
        elements = line.split(" ")
        # print (line)
        elements = [x.strip("\n") for x in elements]
        # print (repr(elements[-1].strip())+repr(elements[-3].strip))
        if elements[-1] == elements[-2]:
            # print ("YWY")
            correctly_predicted_tokens += 1
    f.close()
    return correctly_predicted_tokens


def predict_labels(sess, model, transition_params_trained, dataset,
        epoch_number, stats_graph_folder,
        dataset_filepaths):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model,
            epoch_number, stats_graph_folder,
            transition_params_trained)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[
            dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths


def train_step(sess, dataset, sequence_number, model):
    token_indices_sequence = dataset.token_indices['train'][sequence_number]

    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and \
                np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]

    # dataset
    # feature_list_of_lists=hd.get_features_for_sentence(dataset.adresses[
    # "train"],sequence_number)
    # print (len(feature_list_of_lists))
    # line = linecache.getline(dataset.train_address, sequence_number+1)
    # list_of_list=hd.string_to_list_of_lists(line)
    # print ("SEQENCE NUMBER CHECK")
    # print (sequence_number)
    # print (dataset.label_vector_indices['train'][sequence_number])
    # print (dataset.token_lengths['train'][sequence_number])
    feed_dict = {
        model.input_token_indices: token_indices_sequence,
        model.input_label_indices_vector: dataset.label_vector_indices['train'][
            sequence_number],
        model.input_token_character_indices:
            dataset.character_indices_padded['train'][sequence_number],
        model.input_token_lengths: dataset.token_lengths['train'][
            sequence_number],
        model.input_label_indices_flat: dataset.label_indices['train'][
            sequence_number],
        # ADDED FOR TEST
        model.dropout_keep_prob: 0.5
        #  model.input_features:feature_list_of_lists
        # model.input_label_indices_flat: dataset.label_indices['train'][
        # sequence_number],
    }
    # loss=sess.run([model.loss],feed_dict)
    _, _, loss, accuracy, transition_params_trained = sess.run(
        [model.train_op, model.global_step, model.loss, model.accuracy,
         model.transition_parameters], feed_dict)
    # print loss
    return loss, accuracy, transition_params_trained


def prediction_step(sess, dataset, dataset_type, model, epoch_number,
        results_folder, transition_params_trained,
        use_crf=True):
    print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []

    store_at = results_folder + "epoche_" + str(epoch_number) + ".txt"
    store_at_tes = results_folder + "train/epoche_" + str(epoch_number) + ".txt"
    store_at_valid = results_folder + "valid/epoche_" + str(
        epoch_number) + ".txt"

    # print ("CURRENT DIRECTORY")
    # print (os.getcwd())
    f_store = open(store_at, 'a')
    f_store_train = open(store_at_tes, 'a')
    f_store_valid = open(store_at_valid, 'a')

    prediction_list = []

    for i in range(len(dataset.token_indices[dataset_type])):
        # feature_list_of_lists=hd.get_features_for_sentence(
        # dataset.adresses[dataset_type],i)

        feed_dict = {
            model.input_token_indices: dataset.token_indices[dataset_type][i],
            model.input_token_character_indices:
                dataset.character_indices_padded[dataset_type][i],
            model.input_token_lengths: dataset.token_lengths[dataset_type][i],
            model.input_label_indices_vector:
                dataset.label_vector_indices[dataset_type][i],
            model.input_label_indices_flat: dataset.label_indices[dataset_type][
                i],

            model.dropout_keep_prob: 1
            # model.input_features:feature_list_of_lists
        }
        unary_scores, predictions = sess.run(
            [model.unary_scores, model.predictions], feed_dict)

        if use_crf:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores,
                transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()  # NO CRF ON TOP

        assert len(predictions) == len(dataset.tokens[dataset_type][i])

        prediction_labels = [dataset.index_to_label[prediction] for prediction
                             in predictions]
        gold_labels = dataset.labels[dataset_type][i]

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

        prediction_list.append(prediction_labels)

        if dataset != 'deploy':
            for prediction, token, gold_label in zip(prediction_labels,
                    dataset.tokens[dataset_type][i], gold_labels):
                results = (
                        token + " " + "true " + gold_label + " " +
                        prediction)

                if dataset_type == "test":
                    f_store.write(results + "\n")

                if dataset_type == "train":
                    f_store_train.write(results + "\n")

                if dataset_type == "valid":
                    f_store_valid.write(results + "\n")

            if dataset_type == "test":
                f_store.write("\n")
            if dataset_type == "train":
                f_store_train.write("\n")
            if dataset_type == "valid":
                f_store_valid.write("\n")

    if dataset_type == 'deploy':
        return prediction_list
        # f_store.write("EPOCHE END" + "\n")
    _, _, _, _, _, _ = remap_labels(all_predictions, all_y_true,
        dataset)
    # print (sklearn.metrics.classification_report(new_y_true, new_y_pred,
    # digits=4, labels=new_label_indices, target_names=new_label_names))

    f_store.close()
    f_store_train.close()
    # CONL
    conll_evaluation_script = os.path.join('.', 'conlleval')
    conll_output_filepath = '{0}_conll_evaluation.txt'.format(store_at)
    shell_read = store_at

    if dataset_type == "train":
        print("TRAIN")
        conll_output_filepath = '{0}_conll_evaluation.txt'.format(store_at_tes)
        shell_read = store_at_tes

    if dataset_type == "valid":
        print("VALID")
        conll_output_filepath = '{0}_conll_evaluation.txt'.format(
            store_at_valid)
        shell_read = store_at_valid

    shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script,
        shell_read, conll_output_filepath)
    # print('shell_command: {0}'.format(shell_command))
    os.system(shell_command)
    conll_parsed_output = helper_dataset.get_parsed_conll_output(
        conll_output_filepath)
    # print ("Test F1")
    # print (conll_parsed_output['all']['f1'])

    return (conll_parsed_output['all']['f1']), prediction_list
