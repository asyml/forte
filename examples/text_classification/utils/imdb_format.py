# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
"""Read all data in IMDB and merge them to a csv file."""

import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Process input and output")
parser.add_argument('--raw_data_dir', help='raw data directory')
parser.add_argument('--output_dir', help='output directory')
parser.add_argument('--train_id_path', help='path of id list')
args = parser.parse_args()


def dump_raw_data(contents, file_path):
    with open(file_path, "w", encoding="utf-8") as ouf:
        writer = csv.writer(ouf, delimiter="\t", quotechar="\"")
        for line in contents:
            writer.writerow(line)


def clean_web_text(st):
    """clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
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
    st = st.replace("\\n", " ")
    # st = st.replace("\\", " ")
    # while "  " in st:
    #   st = st.replace("  ", " ")
    return st


def load_data_by_id(sub_set, id_path):
    with open(id_path, encoding="utf-8") as inf:
        id_list = inf.readlines()
    contents = []
    for example_id in id_list:
        example_id = example_id.strip()
        label = example_id.split("_")[0]
        file_path = os.path.join(args.raw_data_dir, sub_set, label, example_id[len(label) + 1:])
        with open(file_path, encoding="utf-8") as inf:
            st_list = inf.readlines()
            assert len(st_list) == 1
            st = clean_web_text(st_list[0].strip())
            contents += [(st, label, example_id)]
    return contents


def load_all_data(sub_set):
    contents = []
    for label in ["pos", "neg", "unsup"]:
        data_path = os.path.join(args.raw_data_dir, sub_set, label)
        if not os.path.exists(data_path):
            continue
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            with open(file_path, encoding="utf-8") as inf:
                st_list = inf.readlines()
                assert len(st_list) == 1
                st = clean_web_text(st_list[0].strip())
                example_id = "{}_{}".format(label, filename)
                contents += [(st, label, example_id)]
    return contents


def main():
    # load train
    header = ["content", "label", "id"]
    contents = load_data_by_id("train", args.train_id_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dump_raw_data(
        [header] + contents,
        os.path.join(args.output_dir, "train.csv"),
        )
    # load test
    contents = load_all_data("test")
    dump_raw_data(
        [header] + contents,
        os.path.join(args.output_dir, "test.csv"),
        )


if __name__ == "__main__":
    main()
