# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the Data Loading Pipeline for Sentence Classifier Task from:
    `https://github.com/google-research/bert/blob/master/run_classifier.py`
"""

import copy
import os
import csv
import logging
import math
import random

import numpy as np
import texar.torch as tx


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0:
                    lines.append(line)
        return lines


class IMDbProcessor(DataProcessor):
    """Processor for the IMDb data set."""

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                           quotechar='"'), "train")

    def get_dev_examples(self, raw_data_dir):
        """The IMDB dataset does not have a dev set so we just use test set"""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "test.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.csv"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar='"'), "unsup_in", skip_unsup=False)

    def get_unsup_aug_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.csv"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train_aug.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_labels(self):
        """See base class."""
        return ["pos", "neg"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        print(len(lines))
        for (i, line) in enumerate(lines):
            if i == 0 or len(line) == 1:  # newline
                continue
            if skip_unsup and line[-2] == "unsup":
                continue
            # Original UDA implementation
            # if line[-2] == "unsup" and len(line[0]) < 500:
                # tf.logging.info("skipping short samples:{:s}".format(line[0]))
                # continue
            guid = "%s-%s" % (set_type, line[-1])
            text_a = " ".join(line[:-2])
            label = line[-2]
            if label not in ["pos", "neg", "unsup"]:
                print(line)
            examples.append(InputExample(guid=guid, text_a=text_a,
                             text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 25000

    def get_dev_size(self):
        return 25000


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    r"""Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids, segment_ids, input_mask = \
        tokenizer.encode_text(text_a=example.text_a,
                              text_b=example.text_b,
                              max_seq_length=max_seq_length)

    label_id = label_map[example.label]

    # here we disable the verbose printing of the data
    if ex_index < 0:
        logging.info("*** Example ***")
        logging.info("guid: %s", example.guid)
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_ids length: %d", len(input_ids))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)", example.label, label_id)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
    return feature


def convert_examples_to_features_and_output_to_files(
        examples, label_list, max_seq_length, tokenizer, output_file,
        feature_types):
    r"""Convert a set of `InputExample`s to a pickled file."""

    with tx.data.RecordData.writer(output_file, feature_types) as writer:
        for (ex_index, example) in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)

            features = {
                "input_ids": feature.input_ids,
                "input_mask": feature.input_mask,
                "segment_ids": feature.segment_ids,
                "label_ids": feature.label_id
            }
            writer.write(features)


def convert_unsup_examples_to_features_and_output_to_files(
        examples, aug_examples, label_list, max_seq_length, tokenizer, output_file,
        feature_types):
    r"""Convert a set of `InputExample`s and the augmented examples
        to a pickled file.
    """

    with tx.data.RecordData.writer(output_file, feature_types) as writer:
        print(len(examples), "unsup examples")
        print(len(aug_examples), "augmented unsup examples")
        assert(len(examples) == len(aug_examples))
        for (ex_index, (example, aug_example)) in enumerate(zip(examples, aug_examples)):
            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)
            aug_feature = convert_single_example(ex_index, aug_example, label_list,
                                                 max_seq_length, tokenizer)

            features = {
                "input_ids": feature.input_ids,
                "input_mask": feature.input_mask,
                "segment_ids": feature.segment_ids,
                "label_ids": feature.label_id,
                "aug_input_ids": aug_feature.input_ids,
                "aug_input_mask": aug_feature.input_mask,
                "aug_segment_ids": aug_feature.segment_ids,
                "aug_label_ids": aug_feature.label_id,
            }
            writer.write(features)


def replace_with_length_check(
        ori_text, new_text,
        use_min_length,
        use_max_length_diff_ratio):
    """Use new_text if the text length satisfies several constraints."""
    if len(ori_text) < use_min_length or len(new_text) < use_min_length:
        if random.random() < 0.001:
            print("not replacing due to short text: \n\tori: {:s}\n\tnew: {:s}\n".format(
                            ori_text,
                            new_text))
        return ori_text
    length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
    if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
        if random.random() < 0.001:
            print("not replacing due to too different text length:\n"
                     "\tori: {:s}\n\tnew: {:s}\n".format(
                             ori_text,
                             new_text))
        return ori_text
    return new_text


def back_translation(examples, back_translation_file, data_total_size):
    """Run back translation."""
    use_min_length = 10
    use_max_length_diff_ratio = 0.5

    text_per_example = 1

    with open(back_translation_file, encoding='utf-8') as inf:
        paraphrases = inf.readlines()
    for i in range(len(paraphrases)):
        paraphrases[i] = paraphrases[i].strip()
    assert len(paraphrases) == data_total_size

    aug_examples = []
    aug_cnt = 0
    for i in range(len(examples)):
        ori_example = examples[i]
        text_a = replace_with_length_check(
                ori_example.text_a,
                paraphrases[i * text_per_example],
                use_min_length,
                use_max_length_diff_ratio,
                )
        if text_a == paraphrases[i * text_per_example]:
            aug_cnt += 1
        if ori_example.text_b is not None:
            text_b = replace_with_length_check(
                    ori_example.text_b,
                    paraphrases[i * text_per_example + 1],
                    use_min_length,
                    use_max_length_diff_ratio,
                    )
        else:
            text_b = None

        example = InputExample(
                guid=ori_example.guid,
                text_a=text_a,
                text_b=text_b,
                label=ori_example.label)
        aug_examples += [example]
        if i % 10000 == 0:
            print("processing example # {:d}".format(i))
    logging.info("applied back translation for {:.1f} percent of data".format(
            aug_cnt * 1. / len(examples) * 100))
    logging.info("finishing running back translation augmentation")
    return aug_examples


def prepare_record_data(processor, tokenizer,
                        data_dir, max_seq_length, output_dir,
                        feature_types, unsup_feature_types=None, sup_size_limit=None, unsup_bt_file=None):
    r"""Prepare record data.
    Args:
        processor: Data Preprocessor, which must have get_labels,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the pickled file in.
        feature_types: The original type of the feature.
    """
    label_list = processor.get_labels()

    train_file = os.path.join(output_dir, "train.pkl")
    if not os.path.isfile(train_file):
        train_examples = processor.get_train_examples(data_dir)
        if sup_size_limit is not None:
            train_examples = get_data_by_size_lim(train_examples, processor, sup_size_limit)
        convert_examples_to_features_and_output_to_files(
            train_examples, label_list, max_seq_length,
            tokenizer, train_file, feature_types)

    eval_file = os.path.join(output_dir, "eval.pkl")
    if not os.path.isfile(eval_file):
        eval_examples = processor.get_dev_examples(data_dir)
        convert_examples_to_features_and_output_to_files(
            eval_examples, label_list,
            max_seq_length, tokenizer, eval_file, feature_types)

    unsup_file = os.path.join(output_dir, "unsup.pkl")
    if not os.path.isfile(unsup_file):
        unsup_label_list = label_list + ["unsup"]
        unsup_examples = processor.get_unsup_examples(data_dir, "unsup_in")
        unsup_aug_examples = copy.deepcopy(unsup_examples)
        unsup_aug_examples = back_translation(unsup_aug_examples, unsup_bt_file, len(unsup_aug_examples))
        convert_unsup_examples_to_features_and_output_to_files(
            unsup_examples, unsup_aug_examples, unsup_label_list,
            max_seq_length, tokenizer, unsup_file, unsup_feature_types)


def get_data_by_size_lim(train_examples, processor, sup_size):
    """Deterministicly get a dataset with only sup_size examples."""
    # Assuming sup_size < number of labeled data and
    # that there are same number of examples for each category
    assert sup_size % len(processor.get_labels()) == 0
    per_label_size = sup_size // len(processor.get_labels())
    per_label_examples = {}
    for i in range(len(train_examples)):
        label = train_examples[i].label
        if label not in per_label_examples:
            per_label_examples[label] = []
        per_label_examples[label] += [train_examples[i]]

    for label in processor.get_labels():
        assert len(per_label_examples[label]) >= per_label_size, (
            "label {} only has {} examples while the limit"
            "is {}".format(label, len(per_label_examples[label]), per_label_size))

    new_train_examples = []
    for i in range(per_label_size):
        for label in processor.get_labels():
            new_train_examples += [per_label_examples[label][i]]
    train_examples = new_train_examples
    return train_examples


def prepare_data(pretrained_model_name, config_data, data_dir):
    """Prepares data.
    """
    logging.info("Loading data")

    processor = IMDbProcessor()

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=pretrained_model_name)

    prepare_record_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=config_data.max_seq_length,
        output_dir=data_dir,
        feature_types=config_data.feature_types,
        unsup_feature_types=config_data.unsup_feature_types,
        sup_size_limit=config_data.num_train_data,
        unsup_bt_file=config_data.unsup_bt_file,
    )
