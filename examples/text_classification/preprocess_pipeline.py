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

from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.data.readers import LargeMovieReader
from forte.pipeline import Pipeline
from forte.utils.utils_io import maybe_create_dir
from ft.onto.base_ontology import Document


def main():
    pipeline = Pipeline[MultiPack]()
    reader = LargeMovieReader()
    pipeline.set_reader(reader)
    pipeline.add(MultiPackBoxer())

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

    for split in ["train", "test"]:
        with open(output_file_path[split], "w", encoding="utf-8")\
            as output_file:
            output_file.write("\t".join(["content", "label", "id"]) + "\n")
            for label in set_labels[split]:
                data_packs = \
                    pipeline.process_dataset(
                        os.path.join(input_file_path[split], label))
                for pack in data_packs:
                    example_id = pack.get_pack('default').pack_name
                    for pack_name in pack.pack_names:
                        p = pack.get_pack(pack_name)
                        for doc in p.get(Document):
                            output_file.write(
                                "\t".join([doc.text, label, example_id]) + "\n")


if __name__ == "__main__":
    main()
