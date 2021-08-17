# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import importlib
import logging

import torch
import texar.torch as tx

import data_utils

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file-name",
    default="data/source/dataset.txt",
    help="Data directory to read the files from",
)
parser.add_argument(
    "--output-dir",
    default="data/",
    help="Output directory to write the pickled files",
)
parser.add_argument(
    "--config-data", default="config_data", help="Configuration file"
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_data = importlib.import_module(args.config_data)


def main():
    train_data = []
    eval_data = []
    test_data = []

    train, eval, test = data_utils.split_train_eval_test(
        file_name=args.file_name
    )

    logging.info("Gather data using history and negative samples...")

    # train data
    conversation_with_history = data_utils.create_dataset_with_history(
        train, num_line=2
    )

    (
        contexts,
        responses,
        negative_responses,
    ) = data_utils.generate_negative_examples(conversation_with_history)

    for context, response, negative_response in zip(
        contexts, responses, negative_responses
    ):
        train_data.append((context, response, "1"))
        train_data.append((context, negative_response, "0"))

    # eval data
    conversation_with_history = data_utils.create_dataset_with_history(
        eval, num_line=2
    )

    (
        contexts,
        responses,
        negative_responses,
    ) = data_utils.generate_negative_examples(conversation_with_history)

    for context, response, negative_response in zip(
        contexts, responses, negative_responses
    ):
        eval_data.append((context, response, "1"))
        eval_data.append((context, negative_response, "0"))

    # test data
    conversation_with_history = data_utils.create_dataset_with_history(
        test, num_line=2
    )

    (
        contexts,
        responses,
        negative_responses,
    ) = data_utils.generate_negative_examples(conversation_with_history)

    for context, response, negative_response in zip(
        contexts, responses, negative_responses
    ):
        test_data.append((context, response, "1"))
        test_data.append((context, negative_response, "0"))

    logging.info("Serialize the data...")

    datasets = {"train": train_data, "eval": eval_data, "test": test_data}

    output_files = [
        os.path.join(args.output_dir, "train.pkl"),
        os.path.join(args.output_dir, "eval.pkl"),
        os.path.join(args.output_dir, "test.pkl"),
    ]

    tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")

    max_len = config_data.max_seq_length
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    example_count = {}
    for dataset, output_file in zip(datasets.items(), output_files):
        with tx.data.RecordData.writer(
            output_file, config_data.feature_types
        ) as writer:
            count = 0
            for datum in dataset[1]:
                sentence_a, sentence_b, label = datum[0], datum[1], datum[2]

                # take the last max_len tokens
                a_tokens = tokenizer.map_text_to_id(sentence_a)[-max_len + 2 :]

                b_tokens = tokenizer.map_text_to_id(sentence_b)[-max_len + 2 :]

                sent_a_input_ids = (
                    [tokenizer.map_token_to_id(cls_token)]
                    + a_tokens
                    + [tokenizer.map_token_to_id(sep_token)]
                )

                if len(sent_a_input_ids) < max_len:
                    sent_a_input_ids += [0] * (max_len - len(sent_a_input_ids))
                sent_a_segment_ids = [0] * max_len
                sent_a_seq_len = len(sent_a_input_ids)

                sent_b_input_ids = (
                    [tokenizer.map_token_to_id(cls_token)]
                    + b_tokens
                    + [tokenizer.map_token_to_id(sep_token)]
                )

                if len(sent_b_input_ids) < max_len:
                    sent_b_input_ids += [0] * (max_len - len(sent_b_input_ids))
                sent_b_segment_ids = [0] * max_len
                sent_b_seq_len = len(sent_b_input_ids)

                features = {
                    "sent_a_input_ids": sent_a_input_ids,
                    "sent_a_seq_len": sent_a_seq_len,
                    "sent_a_segment_ids": sent_a_segment_ids,
                    "sentence_a": sentence_a,
                    "sent_b_input_ids": sent_b_input_ids,
                    "sent_b_seq_len": sent_b_seq_len,
                    "sent_b_segment_ids": sent_b_segment_ids,
                    "sentence_b": sentence_b,
                    "label_ids": label,
                }
                count += 1
                writer.write(features)
            example_count[dataset[0]] = count

    logging.info(
        "Summary: # of train examples = %s, # of eval examples = %s, "
        "# of test examples = %s",
        example_count["train"],
        example_count["eval"],
        example_count["test"],
    )


if __name__ == "__main__":
    main()
