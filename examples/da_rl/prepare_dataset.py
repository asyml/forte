# Copyright 2020 The Forte Authors. All Rights Reserved.
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
# sample from IMDB dataset to prepare fewer examples to train the da_rl model.

import csv

train_POS = 40
train_NEG = 40
eval_POS = 5
eval_NEG = 5


def read_train_csv(input_file, train_output_file,
                   eval_output_file, train_pos, train_neg,
                   eval_pos, eval_neg):
    num_pos, num_neg = 0, 0
    num_eval_pos, num_eval_neg = 0, 0

    with open(input_file, mode='r') as fin:
        with open(train_output_file, mode='w') as fout_train:
            with open(eval_output_file, mode='w') as fout_eval:
                csv_reader = csv.reader(fin, delimiter='\t', quotechar='"')
                cvs_writer_train = \
                    csv.writer(fout_train, delimiter='\t', quotechar='"')
                cvs_writer_eval = \
                    csv.writer(fout_eval, delimiter='\t', quotechar='"')

                cvs_writer_train.writerow(["content", "label", "id"])
                cvs_writer_eval.writerow(["content", "label", "id"])

                for row in csv_reader:
                    if num_eval_pos == eval_pos and num_eval_neg == eval_neg:
                        break
                    if len(row) == 0:
                        continue
                    if row[1] == "unsup":
                        continue
                    if row[1] == "neg" and num_eval_neg == eval_neg:
                        continue
                    if row[1] == "pos" and num_eval_pos == eval_pos:
                        continue
                    if row[1] == "neg" and num_neg == train_neg:
                        cvs_writer_eval.writerow(row)
                        num_eval_neg += 1
                    elif row[1] == "pos" and num_pos == train_pos:
                        cvs_writer_eval.writerow(row)
                        num_eval_pos += 1
                    elif row[1] == "neg":
                        cvs_writer_train.writerow(row)
                        num_neg += 1
                    elif row[1] == "pos":
                        cvs_writer_train.writerow(row)
                        num_pos += 1


read_train_csv("data/IMDB/train.csv", "data/IMDB/train_small.csv",
               "data/IMDB/eval_small.csv", train_POS, train_NEG,
               eval_POS, eval_NEG)
