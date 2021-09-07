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

import csv
import importlib
from typing import Iterator, Tuple
from collections import OrderedDict
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import *


__all__ = ["ClassificationDatasetReader"]


class ClassificationDatasetReader(PackReader):
    r"""
    A generic dataset reader class for classification datasets.

    """

    def __init__(self):
        super().__init__()
        self.index2class = None
        self.label2index = None

    def set_up(self):
        assert self.configs.index2class is not None, (
            "User must set index2class to enable"
            " the dataset reader encode labels correctly"
        )
        # set up class names

        # class and index
        if not self.configs.digit_label:
            # initialize class
            self.index2class = self.configs.index2class
            self.label2index = dict(
                [(v, k) for k, v in self.index2class.items()]
            )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()

    def _collect(self, csv_file: str) -> Iterator[Tuple[int, str]]:
        with open(csv_file, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            next(data)
            for line_id, line in enumerate(data):
                yield line_id, line

    def _cache_key_function(self, line_info: Tuple[int, str]) -> str:
        return str(line_info[0])

    def _parse_pack(self, line_info: Tuple[int, str]) -> Iterator[DataPack]:
        line_id, line = line_info
        # content label title
        pack = DataPack()

        # subtext fields must follow the ontology names
        data_fields = self.configs.data_fields
        assert (
            "label" in data_fields
        ), "There must be label data field in reader config."
        assert len(self.configs.data_fields) == len(line), (
            "Data fields provided in config "
            "is not aligned with the actual line info from dataset."
            + str((data_fields, line))
        )
        assert (
            len(self.configs.subtext_fields) > 0
        ), "There must be at least one subtext to reader to select from"

        df_dict = OrderedDict()
        for df, value in zip(data_fields, line):
            df_dict[
                df
            ] = value  # in general, value can be subtext or other types of data

        # it determines the order of concatenation
        subtext_fields = self.configs.subtext_fields
        # get text and subtext indices
        assert set(self.configs.subtext_fields).issubset(
            set(self.configs.data_fields)
        ), "subtext fields must be a subset of data fields"

        text, subtext_indices = generate_text_n_subtext_indices(
            subtext_fields, df_dict
        )
        pack.set_text(text, replace_func=self.text_replace_operation)
        assert df_dict["label"].isdigit() == self.configs.digit_label, (
            "Label format from dataset"
            " is not consistent with the label format from"
            " configs"
        )
        if self.configs.digit_label:
            if self.configs.one_based_index_label:
                class_id = int(df_dict["label"]) - 1
            else:
                class_id = int(df_dict["label"])
        else:
            class_id = self.label2index[df_dict["label"]]
        for subtext_field in subtext_indices.keys():
            path_str, module_str = subtext_field.rsplit(".", 1)
            mod = importlib.import_module(path_str)  # sentence ontology module
            entry_class = getattr(mod, module_str)
            start_idx, end_idx = subtext_indices[subtext_field]
            entry_class(pack, start_idx, end_idx)

        doc = Document(pack, 0, subtext_indices[subtext_fields[-1]][1])
        doc.document_class = [
            self.configs.index2class[class_id]
        ]  # takes names rather than id
        # TODO: add labels in document_class
        # TODO: some datasets have two sets of labels, but it should be controlled by the user

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()
        config.update(
            {
                "data_fields": [
                    "label",
                    "ft.onto.base_ontology.Title",
                    "ft.onto.ag_news.Description",
                ],
                # data fields aligned with columns in dataset
                "index2class": None,
                "subtext_fields": [
                    "ft.onto.base_ontology.Title",
                    "ft.onto.ag_news.Description",
                ],
                # select subtexts to concatenate into text
                "digit_label": True,  # specify whether label in dataset is digit
                "text_label": False,  # either digit label or text label
                "one_based_index_label": True,
                # if it's digit label, whether it's one-based so that reader can adjust it
            }
        )
        return config


def generate_text_n_subtext_indices(subtext_fields, data_fields_dict):
    """
    Retrieve subtext from data fields and concatenate them into text.
    Also, we generate the indices for these subtext accordingly.

    :param subtext_fields:
    :param data_fields_dict:
    :return:
    """
    end = -1
    text = ""
    indices = {}  # a dictionary of (subtext_name: (start_index, end_index) )
    for i, sub_text_field in enumerate(subtext_fields):
        if not i:
            text += "\n"
        text += data_fields_dict[sub_text_field]
        start = end + 1
        end = len(text)
        indices[subtext_fields[i]] = (start, end)
    return text, indices
