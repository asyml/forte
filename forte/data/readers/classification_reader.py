# Copyright 2022 The Forte Authors. All Rights Reserved.
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
from typing import Iterator, Tuple, List, Dict
from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import *

__all__ = ["ClassificationDatasetReader"]


class ClassificationDatasetReader(PackReader):
    r""":class:`ClassificationDatasetReader` is designed to read table-like
    classification datasets that contain input text and digit/text labels.
    There are a couple of values that need to be provided via configuration,
    including `forte_data_fields`, `text_fields`, and `index2class`.

    `forte_data_fields` is a list representing the column headers of the
    dataset. Each element in the `forte_data_fields` list is either a label
    that indicates the body class or a Forte entry type that will be used
    to store the content. Apparently, the list should follow the column order
    of the dataset.

    For example, for amazon polarity dataset,
    (https://huggingface.co/datasets/amazon_polarity), the column names
    are [label, title, content]. We can configure `forte_data_fields`to be
    ['label', 'ft.onto.base_ontology.Title', 'ft.onto.base_ontology.Body'].
    `label` is a special keyword to specify the label/body class column,
    while the latter two ontology types will be used by the reader to store the
    text from the `title` and `content` column respectively.

    `text_fields` is a list of Forte entry types that indicate texts in the
    forte data fields will be kept and concatenated as input text.
    Apparently, it's a subset of `forte_data_fields`.
    For example, if we only want titles and bodies in our input text, we
    specify "text_fields":
    ['ft.onto.base_ontology.Title', 'ft.onto.base_ontology.Body']" in the
    configuration. If titles are not needed, we can also only include
    bodies by specifying
    "text_fields":['ft.onto.base_ontology.Body']" which is very
    flexible in customizing input text.


    `index2class` is a dictionary that the mapping from zero-based
    indices to classes. For example, in amazon polarity dataset, we have two
    classes, negative and positive.
    We can configure `index2class` to be `{0: negative, 1: positive}`.
    sentiment classifications.

    To see a full example, please refer to
    https://github.com/asyml/forte/examples/classification
    """

    def __init__(self):
        super().__init__()
        self._index2class = None
        self._class2index = None

    def set_up(self):
        if self.configs.index2class is None:
            raise ProcessorConfigError(
                "User must set index2class to enable"
                " the dataset reader to encode labels correctly."
                "Current index2class is set to "
                f"{self.configs.index2class}"
            )
        if self.configs.skip_k_starting_lines < 0:
            raise ProcessorConfigError(
                "Current skip_k_starting_lines in configuration is set to "
                f"{self.configs.skip_k_starting_lines}. User must specify an"
                " integer larger or equal to 0."
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
                "There must be a forte data field named 'label'"
                " in the reader config."
            )

        if "ft.onto.base_ontology.Document" in self.configs.forte_data_fields:
            raise ProcessorConfigError(
                "Found `ft.onto.base_ontology.Document` in forte_data_fields "
                "of the reader config. Try use a different ontology to "
                "represent the data field as this ontology is reserved to "
                "store the concatenated text from input."
            )

        if self.configs.text_fields is None:
            self.configs.text_fields = [
                df
                for df in self.configs.forte_data_fields
                if df is not None and df != "label"
            ]
        else:
            if not set(self.configs.text_fields).issubset(
                set(self.configs.forte_data_fields)
            ):
                raise ProcessorConfigError(
                    "text_fields must be a subset of forte_data_fields."
                    f"text_fields: {self.configs.text_fields}"
                    f"forte_data_fields: {self.configs.forte_data_fields}"
                    "Please correct text_fields and forte_data_fields in the"
                    " configuration to satisfy the condition."
                )

    def _collect(  # type: ignore
        self, csv_file: str
    ) -> Iterator[Tuple[int, List[str]]]:
        with open(csv_file, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            if self.configs.skip_k_starting_lines > 0:
                for _ in range(self.configs.skip_k_starting_lines):
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
                "Forte data fields provided in config "
                "doesn't have one-to-one correspondence with the actual data "
                "fields from dataset.\n"
                f"Data fields length: {len(self.configs.forte_data_fields)} \n"
                f"Line length: {len(line)}"
            )
        df_dict = dict(zip(self.configs.forte_data_fields, line))

        # it determines the order of concatenation
        text_fields = self.configs.text_fields

        # get text and ontology indices
        text, input_ontology_indices = generate_text_n_input_ontology_indices(
            text_fields, df_dict
        )
        self.set_text(pack, text)
        if df_dict["label"].isdigit() != self.configs.digit_label:
            dataset_digit_label = df_dict["label"].isdigit()
            raise ProcessorConfigError(
                "Label format from dataset "
                "is not consistent with the label format from configs. \n"
                f"dataset digit label: {dataset_digit_label} \n"
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
        for (
            ontology_field,
            (start_idx, end_idx),
        ) in input_ontology_indices.items():
            path_str, module_str = ontology_field.rsplit(".", 1)
            mod = importlib.import_module(path_str)  # sentence ontology module
            entry_class = getattr(mod, module_str)
            entry_class(pack, start_idx, end_idx)
        # for now, we use Document to store concatenated text and set the class here
        doc = Document(pack, 0, input_ontology_indices[text_fields[-1]][1])
        doc.document_class = [self.configs.index2class[class_id]]

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for classification dataset reader.

        Here:
          - forte_data_fields: these fields provides one-to-one
              correspondence between given original dataset column names and
              labels or forte ontology. For column names without usage,
              user can specify None for them.
          - index2class: a dictionary that maps from zero-based indices to
              classes
          - text_fields: a list of ordered input ontology that
              user want to concatenate into an input text.
          - digit_label:  boolean value that specifies whether label in dataset
              is digit.
          - one_based_index_label: boolean value that specifies if dataset
              provides one-based digit label.
              True for one-based index, false otherwise.
          - skip_k_starting_lines: many datasets' first line are columns
              names, set it to 1 if it's the case. Otherwise set to 0.
              User can also set it to other positive integers to skip
              multiple lines.
        """
        return {
            "forte_data_fields": [
                "label",
                "ft.onto.base_ontology.Title",
                "ft.onto.base_ontology.Body",
            ],
            "index2class": None,
            "text_fields": None,
            "digit_label": True,
            "one_based_index_label": True,
            "skip_k_starting_lines": 1,
        }


def generate_text_n_input_ontology_indices(
    text_fields: List[str], forte_data_fields_dict: Dict[str, str]
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Retrieve ontologies from data fields and concatenate them into text.
    Also, we generate the indices for these ontologies accordingly.

    Args:
        text_fields(List[str]): a list of ontology that needs to be
            concatenated into a input string.
        forte_data_fields_dict(Dict[str, str]): a dictionary with ontology names as keys and
            ontology strings as values.

    Returns:
        Tuple[str, Dict[str, Tuple[int, int]]]: a concatenated text and
            dictionary that keys are forte data entries and values are start
            and end indices of the data entries.
    """
    end = -1
    text = ""
    indices = {}  # a dictionary of (ontology_name: (start_index, end_index) )
    for i, input_onto in enumerate(text_fields):
        text += forte_data_fields_dict[input_onto]
        start = end + 1
        end = len(text)
        if i != len(text_fields) - 1:  # not last input ontology
            text += "\n"
        indices[text_fields[i]] = (start, end)

    return text, indices
