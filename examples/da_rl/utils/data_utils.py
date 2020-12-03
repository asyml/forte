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

import os
import csv
import logging

import texar.torch as tx


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples
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
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines


def clean_web_text(st):
    """clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        # print("before:\n", st)
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1:]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
        # print("after\n", st)
        # print("")
    st = st.replace("\\n", " ")
    st = st.replace("\\", " ")
    # while "  " in st:
    #   st = st.replace("  ", " ")
    return st


class IMDbProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train_small.csv"),
                           quotechar='"'), "train")

    def get_dev_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "eval_small.csv"),
                           quotechar='"'), "dev")

    def get_test_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "test_small.csv"),
                           quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.csv"),
                               quotechar='"'), "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train.csv"),
                               quotechar='"'), "unsup_in", skip_unsup=False)

    def get_labels(self):
        """See base class."""
        return ["pos", "neg"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 0:
                continue
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                # logging.info("skipping short samples:{:s}".format(line[0]))
                continue
            guid = "%s-%s" % (set_type, line[2])
            text_a = line[0]
            label = line[1]
            text_a = clean_web_text(text_a)
            examples.append(InputExample(guid=guid, text_a=text_a,
                             text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 80

    def get_dev_size(self):
        return 10


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
        logging.info("segment_ids: %s",
                     " ".join([str(x) for x in segment_ids]))
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


def prepare_record_data(processor, tokenizer,
                        data_dir, max_seq_length, output_dir,
                        feature_types):
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

    train_examples = processor.get_train_examples(data_dir)
    train_file = os.path.join(output_dir, "train.pkl")
    convert_examples_to_features_and_output_to_files(
        train_examples, label_list, max_seq_length,
        tokenizer, train_file, feature_types)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(output_dir, "eval.pkl")
    convert_examples_to_features_and_output_to_files(
        eval_examples, label_list,
        max_seq_length, tokenizer, eval_file, feature_types)

    test_examples = processor.get_test_examples(data_dir)
    test_file = os.path.join(output_dir, "predict.pkl")
    convert_examples_to_features_and_output_to_files(
        test_examples, label_list,
        max_seq_length, tokenizer, test_file, feature_types)
