#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:38:57 2017

@author: elena
"""
import sklearn.metrics
from examples.Cliner.CliNER.code import helper_dataset


def assess_model(y_pred, y_true, labels, target_names):
    results = {}
    assert len(y_true) == len(y_pred)

    # Classification report
    classification_report = sklearn.metrics.classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        sample_weight=None,
        digits=4)
    results['classification_report'] = classification_report

    # F1 scores
    results['f1_score'] = {}
    for f1_average_style in ['weighted', 'micro', 'macro']:
        results['f1_score'][f1_average_style] = sklearn.metrics.f1_score(
            y_true, y_pred, average=f1_average_style, labels=labels) * 100
    results['f1_score']['per_label'] = [
        x * 100 for x in sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels)[2].tolist()
    ]
    results['accuracy_score'] = sklearn.metrics.accuracy_score(y_true,
                                                               y_pred) * 100

    print(results['classification_report'])
    print(results['f1_score']['per_label'])

    return results


def remap_labels(y_pred, y_true, dataset, evaluation_mode='bio'):
    '''
    y_pred: list of predicted labels
    y_true: list of gold labels
    evaluation_mode: 'bio', 'token', or 'binary'

    Both y_pred and y_true must use label indices and names specified in the
    dataset
#     (dataset.unique_label_indices_of_interest,
dataset.unique_label_indices_of_interest).
    '''
    all_unique_labels = dataset.unique_labels
    if evaluation_mode == 'bio':
        # sort label to index
        new_label_names = all_unique_labels[:]
        new_label_names.remove('O')
        new_label_names.sort(
            key=lambda x: (helper_dataset.remove_bio_from_label_name(x), x))
        new_label_names.append('O')
        new_label_indices = list(range(len(new_label_names)))
        new_label_to_index = dict(zip(new_label_names, new_label_indices))

        remap_index = {}
        for i, label_name in enumerate(new_label_names):
            label_index = dataset.label_to_index[label_name]
            remap_index[label_index] = i

    else:
        raise ValueError("At this point only 'bio' is accepted")

    new_y_pred = [remap_index[label_index] for label_index in y_pred]
    new_y_true = [remap_index[label_index] for label_index in y_true]

    new_label_indices_with_o = new_label_indices[:]
    new_label_names_with_o = new_label_names[:]
    new_label_names.remove('O')
    new_label_indices.remove(new_label_to_index['O'])

    return new_y_pred, new_y_true, new_label_indices, new_label_names, \
           new_label_indices_with_o, new_label_names_with_o
