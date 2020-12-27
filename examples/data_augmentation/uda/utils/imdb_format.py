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
"""Read all data in IMDB and merge them to a csv file."""
import os
import csv
import json

from forte.data.multi_pack import MultiPack
from forte.data.readers import LargeMovieReader
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.utils.utils_io import maybe_create_dir
from ft.onto.base_ontology import Document, Sentence


def main():
    pipeline = Pipeline[MultiPack]()
    reader = LargeMovieReader()
    pipeline.set_reader(reader)
    pipeline.add(NLTKSentenceSegmenter())

    pipeline.initialize()

    dataset_path = "data/IMDB_raw/aclImdb/"
    input_file_path = {
        "train": os.path.join(dataset_path, "train"),
        "test": os.path.join(dataset_path, "test")
    }
    output_path = "data/IMDB/"
    maybe_create_dir(output_path)
    output_file_path = {
        "train": os.path.join(output_path, "train.csv"),
        "test": os.path.join(output_path, "test.csv")
    }
    set_labels = {
        "train": ["pos", "neg", "unsup"],
        "test": ["pos", "neg"],
    }

    back_trans_data_path = "back_trans/"
    maybe_create_dir(back_trans_data_path)
    split_sent_output_path = os.path.join(back_trans_data_path, "train_split_sent.txt")
    doc_len_path = os.path.join(back_trans_data_path, "train_doc_len.json")

    sents = []
    doc_lens = []
    for split in ["train", "test"]:
        with open(output_file_path[split], "w", encoding="utf-8")\
            as output_file:
            writer = csv.writer(output_file, delimiter="\t", quotechar="\"")
            writer.writerow(["content", "label", "id"])
            for label in set_labels[split]:
                data_packs = \
                    pipeline.process_dataset(
                        os.path.join(input_file_path[split], label))
                for pack in data_packs:
                    example_id = pack.pack_name
                    for doc in pack.get(Document):
                        writer.writerow(
                            [doc.text.strip(), label, example_id])
                    if split == "train":
                        doc_len = 0
                        for sent in pack.get(Sentence):
                            sents.append(sent.text)
                            doc_len += 1
                        doc_lens.append(doc_len)
    
    with open(split_sent_output_path, "w", encoding="utf-8") as output_file:
        for sent in sents:
            output_file.write(sent + "\n")
    with open(doc_len_path, "w", encoding="utf-8") as output_file:
        json.dump(doc_lens, output_file)


if __name__ == "__main__":
    main()
