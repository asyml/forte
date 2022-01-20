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
from typing import Iterator, Tuple, List
from collections import OrderedDict
from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import *


__all__ = ["ClassificationDatasetReader"]


class ClassificationDatasetReader(PackReader):
    r"""class:`ClassificationDatasetReader` is designed to read text
    classification dataset that contains input text and digit/text labels.

    User must specify the mapping from input data fields to ontologies or
    labels and it should be one-to-one correspondence. This configuration help
    the class generate labels and initialize data fields within corresponding
    ontologies wrappers. For example, for amazon polarity dataset,
    https://huggingface.co/datasets/amazon_polarity, the original data fields
    are [label, title, content]. We want to identify them as either class
    label or specific ontologies by specifying
    "forte_data_fields: ['label', 'ft.onto.base_ontology.Title',
    'ft.onto.ag_news.Description']" in the configuration.

    User must also specify ontologies that will be concatenate as input text
    in a list. The number of input ontologies and the order of concatenation
    can be customized in the list. For example, if we only want titles and
    descriptions in our input text, we specify "input_ontologies":
    ['ft.onto.base_ontology.Title', 'ftx.onto.ag_news.Description']" in the
    configuration. We can also only include descriptions by specifying
    "input_ontologies":['ftx.onto.ag_news.Description']" which is very
    flexible in customizing input text.


    User must also specify the mapping from zero-based indices to classes.
    For example, `index2class: {0: negative, 1: positive}` for polarity
    sentiment classifications.

    To see a full example, please refer to
    https://github.com/asyml/forte/examples/classification
    """

    def __init__(self):
        super().__init__()
        self._index2class = None
        self._class2index = None

    def set_up(self):
        assert self.configs.index2class is not None, (
            "User must set index2class to enable"
            " the dataset reader to encode labels correctly."
        )

        # class and index
        if not self.configs.digit_label:
            # initialize class
            self._index2class = self.configs.index2class
            self._class2index = {v: k for k, v in self._index2class.items()}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()
        if "label" not in self.configs.forte_data_fields:
            raise ProcessorConfigError(
                "There must be data field named 'label' in reader config."
            )

        if not self.configs.input_ontologies:
            raise ProcessorConfigError(
                "There must be at least one ontology field "
                + "to reader to select from."
            )

        if not set(self.configs.input_ontologies).issubset(
            set(self.configs.forte_data_fields)
        ):
            raise ProcessorConfigError(
                "ontology fields must be a subset of data fields"
            )

    def _collect(self,
                 csv_file: str) -> Iterator[Tuple[int, List[str]]]:
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

        if len(self.configs.forte_data_fields) != len(line):
            raise ProcessorConfigError(
                "Data fields provided in config "
                "is not aligned with the actual line info from dataset.\n"
                f"Data fields length: {len(self.configs.forte_data_fields)} \n"
                f"Line length: {len(line)}"
            )

        df_dict = OrderedDict()
        for df, value in zip(self.configs.forte_data_fields, line):
            if df is not None:
                df_dict[
                    df
                ] = value  # in general, value can be ontology or other types of data

        # it determines the order of concatenation
        input_ontologies = self.configs.input_ontologies

        # get text and ontology indices
        text, ontology_indices = generate_text_n_ontology_indices(
            input_ontologies, df_dict
        )
        self.set_text(pack, text)
        if df_dict["label"].isdigit() != self.configs.digit_label:
            dataset_digit_label = df_dict["label"].isdigit()
            raise ProcessorConfigError(
                "Label format from dataset "
                "is not consistent with the label format from configs. \n"
                f"dataset digit label: {dataset_digit_label}"
                f"config digit label: {self.configs.digit_label}",
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
        # initialize all ontologies instances as data pack module
        # to add corresponding variables and functions
        for ontology_field, (start_idx, end_idx) in ontology_indices.items():
            path_str, module_str = ontology_field.rsplit(".", 1)
            mod = importlib.import_module(path_str)  # sentence ontology module
            entry_class = getattr(mod, module_str)
            entry_class(pack, start_idx, end_idx)
        # for now, we use document to store concatenated text and set the class here
        doc = Document(pack, 0, ontology_indices[input_ontologies[-1]][1])
        doc.document_class = [self.configs.index2class[class_id]]

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for classification dataset reader.

        Here:
            - forte_data_fields: these fields provides one-to-one
            correspondence between given original dataset data fields and
            labels/ontologies. For data fields without usage, user can
            specify None for them.
            - index2class: a dictionary that maps from zero-based indices to
                classes
            - input_ontologies: a list of ordered input ontologies that
                user want to concatenate into an input text.
            - digit_label:  boolean value that specifies whether label in dataset is digit.
            - one_based_index_label: boolean value that specifies if dataset
                provides one-based digit label.
                True for one-based index, false otherwise.
            - skip_first_line: many datasets' first line are columns names,
                set this config to True if it's the case.
        """
        return {
            "forte_data_fields": [
                "label",
                "ft.onto.base_ontology.Title",
                "ftx.onto.ag_news.Description",
            ],
            "index2class": None,
            "input_ontologies": [
                "ft.onto.base_ontology.Title",
                "ftx.onto.ag_news.Description",
            ],
            "digit_label": True,
            "one_based_index_label": True,
            "skip_first_line": True,
        }


def generate_text_n_ontology_indices(input_ontologies, forte_data_fields_dict):
    """
    Retrieve ontologies from data fields and concatenate them into text.
    Also, we generate the indices for these ontologies accordingly.

    Args:
        input_ontologies: a list of ontology that needs to be concatenated into a input string.
        forte_data_fields_dict: a dictionary with ontology names as key and
            ontology string as value.
    """
    end = -1
    text = ""
    indices = {}  # a dictionary of (ontology_name: (start_index, end_index) )
    for i, sub_text_field in enumerate(input_ontologies):
        if not i:
            text += "\n"
        text += forte_data_fields_dict[sub_text_field]
        start = end + 1
        end = len(text)
        indices[input_ontologies[i]] = (start, end)
    return text, indices
