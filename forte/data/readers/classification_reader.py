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
# mypy: ignore-errors
import csv
import importlib
from typing import Iterator, Tuple, List
from collections import OrderedDict
from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import *


__all__ = ["ClassificationDatasetReader"]


class ClassificationDatasetReader(PackReader):
    r"""
    A generic dataset reader class for classification dataset.
    In general, it works with any classification tasks that
    require concatenating a sequence of strings from dataset and predicting a class.


    The user must specify the index2label mapping, data fields and subtext fields
    and other parameters in configs based on their tasks in order to initialize
    the dataset reader properly.
    Index2label mapping is a dictionary with indices as keys and string labels as values,
    it helps encoding and decoding string labels.
    Data fields is a list of ontology paths representing data fields in the dataset.
    They must have the same length and align with each other.
    Subtext fields is also a list of ontology paths that represent data fields that
     will be concatenated into input strings in the same order as the list.
    User can select an arbitrary sequence of subtexts to concatenate
     as long as there is at least one subtext in the sequence.
     The number and the order of subtexts can be customized depending on the use cases.



    The first line of dataset usually specifies column names of data fields,
    and the user needs to write a list of data ontology paths.
    For example, in Amazon review sentiment dataset, the first line specifies [content, label,
     title].
    By checking the actual data, except for label, we must find their corresponding ontology names
    and its relative path in the given or customized ontology.
    Here we define data_fields = ["label", "ft.onto.base_ontology.Title",
     "ft.onto.ag_news.Description"]
    And we want both Title and Description to be included in input string, so we define
    subtext fields = [ "ft.onto.base_ontology.Title",  "ft.onto.ag_news.Description"].
    To see a full example, please refer to
    https://github.com/asyml/forte/examples/classification
    """

    def __init__(self):
        super().__init__()
        self._index2class = None
        self._class2index = None

    def set_up(self):
        assert self.configs.index2class is not None, (
            "User must set _index2class to enable"
            " the dataset reader to encode labels correctly"
        )

        # class and index
        if not self.configs.digit_label:
            # initialize class
            self._index2class = self.configs.index2class
            self._class2index = {v: k for k, v in self._index2class.items()}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()
        if "label" not in self.configs.data_fields:
            raise ProcessorConfigError(
                "There must be data field named 'label' in reader config."
            )

        if not self.configs.subtext_fields:
            raise ProcessorConfigError(
                "There must be at least one subtext field "
                + "to reader to select from."
            )

        if not set(self.configs.subtext_fields).issubset(
            set(self.configs.data_fields)
        ):
            raise ProcessorConfigError(
                "subtext fields must be a subset of data fields"
            )

    def _collect(self, csv_file: str) -> Iterator[Tuple[int, str]]:
        with open(csv_file, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            if self.configs.skip_first_line:
                next(data)
            for line_id, line in enumerate(data):
                yield line_id, line

    def _cache_key_function(self, line_info: Tuple[int, List[str]]) -> str:
        return str(line_info[0])

    def _parse_pack(
        self, line_info: Tuple[int, List[str]]
    ) -> Iterator[DataPack]:
        line_id, line = line_info
        # content label title
        pack = DataPack()

        # subtext fields must follow the ontology names
        data_fields = self.configs.data_fields

        if len(self.configs.data_fields) != len(line):
            raise ProcessorConfigError(
                "Data fields provided in config "
                "is not aligned with the actual line info from dataset.\n"
                "Data fields length: "
                + str(len(self.configs.data_fields))
                + "\n",
                "Line length: " + str(len(line)),
            )

        df_dict = OrderedDict()
        for df, value in zip(data_fields, line):
            df_dict[
                df
            ] = value  # in general, value can be subtext or other types of data

        # it determines the order of concatenation
        subtext_fields = self.configs.subtext_fields

        # get text and subtext indices
        text, subtext_indices = generate_text_n_subtext_indices(
            subtext_fields, df_dict
        )
        self.set_text(pack, text)
        if df_dict["label"].isdigit() != self.configs.digit_label:
            raise ProcessorConfigError(
                "Label format from dataset "
                "is not consistent with the label format from configs. \n"
                + "dataset digit label status: "
                + str(df_dict["label"].isdigit())
                + "\n"
                "config digit label status: " + str(self.configs.digit_label)
            )

        if self.configs.digit_label:
            if self.configs.one_based_index_label:
                # adjust label encoding if dataset digit label is one-based
                class_id = int(df_dict["label"]) - 1
            else:
                class_id = int(df_dict["label"])
        else:
            # decode string label from dataset into digit label
            class_id = self._class2index[df_dict["label"]]
        # initialize all subtexts instances as data pack module
        # to add corresponding variables and functions
        for subtext_field, (start_idx, end_idx) in subtext_indices.items():
            path_str, module_str = subtext_field.rsplit(".", 1)
            mod = importlib.import_module(path_str)  # sentence ontology module
            entry_class = getattr(mod, module_str)
            entry_class(pack, start_idx, end_idx)
        # for now, we use document to store concatenated text and set the class here
        doc = Document(pack, 0, subtext_indices[subtext_fields[-1]][1])
        doc.document_class = [self.configs.index2class[class_id]]

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
                "skip_first_line": True,
            }
        )
        return config


def generate_text_n_subtext_indices(subtext_fields, data_fields_dict):
    """
    Retrieve subtexts from data fields and concatenate them into text.
    Also, we generate the indices for these subtexts accordingly.

    Args:
        subtext_fields: a list of ontology that needs to be concatenated into a input string
        data_fields_dict: a dictionary with subtext names as key and subtext string as value.
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
