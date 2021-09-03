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
"""
The reader that reads AG News data into Datapacks.
"""
import csv
import sys
import importlib
from typing import Iterator, Tuple
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.ag_news import Description
from ft.onto.base_ontology import *
from collections import OrderedDict

__all__ = [
    "AGNewsReader",
    "AmazonPolarityReader",
    "Banking77",
    "ClassificationDatasetReader"
]

class ClassificationDatasetReader(PackReader):
    r"""

    :param PackReader:
    :return:
    """
    def __init__(self):
        super().__init__()

    def set_up(self):
        # set up class names

        # class and index
        if not self.configs.digit_label:
            # initialize class
            self.index2class = self.configs.index2class
            self.label2index = dict([(v, k) for k, v in self.index2class.items()])

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
        assert "label" in data_fields, "There must be label data field in reader config."
        assert len(self.configs.data_fields) == len(line), "Data fields provided in config " \
                                                          "is not aligned with the actual line info from dataset." \
                                                          + str((data_fields, line))
        assert len(self.configs.subtext_fields) > 0, "There must be at least one subtext to reader to select from"

        df_dict = OrderedDict()
        for df, value in zip(data_fields, line):
            df_dict[df] = value  # in general, value can be subtext or other types of data

        # it determines the order of concatenation
        subtext_fields = self.configs.subtext_fields
        # get text and subtext indices
        assert set(self.configs.subtext_fields) \
            .issubset(set(self.configs.data_fields)), "subtext fields must be a subset of data fields"

        text, subtext_indices = generate_text_n_subtext_indices(subtext_fields, df_dict)
        pack.set_text(text, replace_func=self.text_replace_operation)
        assert df_dict["label"].isdigit() == self.configs.digit_label, "Label format from dataset" \
                                                                       " is not consistent with the label format from" \
                                                                       " configs"
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
        doc.document_class = [self.configs.index2class[class_id]]  # takes names rather than id
        # TODO: add labels in document_class
        # TODO: some datasets have two sets of labels, but it should be controlled by the user

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()
        config.update({
            "data_fields": ["label", "ft.onto.base_ontology.Title", "ft.onto.ag_news.Description"],  # data fields aligned with columns in dataset
            "index2class": {  # zero-based index2label mapping
                0: "negative",
                1: "positive"
            },
            "subtext_fields": ["ft.onto.base_ontology.Title", "ft.onto.ag_news.Description"],  # select subtexts to concatenate into text
            "digit_label": True,  # specify whether label in dataset is digit
            "text_label": False,  # either digit label or text label
            "one_based_index_label": True,  # if it's digit label, whether it's one-based so that reader can adjust it
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


def generate_text_n_anchor(subtext_names, *args):
    """
    A generic method to generate start and end indices for every subtext parameters, such as titles, content.

    :param subtext_names: variable names for subtext
    :param kwargs: subtext
    :return: (text string, a dictionary of {subtext_name: (start_index, end_index) }
    """
    end = -1
    text = ""
    indices = {}  # a dictionary of (subtext_name: (start_index, end_index) )
    for i, sub_text in enumerate(args):
        if not i:
            text += "\n"
        text += sub_text
        start = end + 1
        end = len(text)
        indices[subtext_names[i]] = (start, end)
    return text, indices


class AmazonPolarityReader(PackReader):
    r"""
    Reader for amazon polarity dataset. Texts are customers' reviews, and labels are their sentiments.

    Link for download: https://drive.google.com/u/0/uc?id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM&export=download
    """

    def _collect(self, csv_file: str) -> Iterator[Tuple[int, str]]:
        with open(csv_file, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            next(data)
            for line_id, line in enumerate(data):
                yield line_id, line

    def _cache_key_function(self, line_info: object) -> object:
        return str(line_info[0])

    def _parse_pack(self, line_info: Tuple[int, str]) -> Iterator[DataPack]:
        line_id, line = line_info
        # content label title
        pack = DataPack()
        label, title, content = line

        pack.pack_name = line_id
        text, indices = generate_text_n_anchor(["title", "content"], title, content)
        pack.set_text(text, replace_func=self.text_replace_operation)
        class_id = int(label) - 1

        doc = Document(pack, 0, indices["content"][1])
        doc.document_class = [self.configs.candidate_labels[class_id]]
        Title(pack, 0, indices["title"][1])
        # TODO: maybe another generic name for Description (mainSubText)
        Description(pack, indices["content"][0], indices["content"][1])

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()

        config.update(
            {
                "candidate_labels": {
                    0: "negative",
                    1: "positive"

                }
            }
        )
        return config


class Banking77(PackReader):
    r"""
    Reader for banking77 dataset. Texts are customers' requests and labels are customers' fine-grained intents.

    Train dataset link: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    Test dataset link: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    """

    class_list = ["activate_my_card",
                  "age_limit",
                  "apple_pay_or_google_pay",
                  "atm_support",
                  "automatic_top_up",
                  "balance_not_updated_after_bank_transfer",
                  "balance_not_updated_after_cheque_or_cash_deposit",
                  "beneficiary_not_allowed",
                  "cancel_transfer",
                  "card_about_to_expire",
                  "card_acceptance",
                  "card_arrival",
                  "card_delivery_estimate",
                  "card_linking",
                  "card_not_working",
                  "card_payment_fee_charged",
                  "card_payment_not_recognised",
                  "card_payment_wrong_exchange_rate",
                  "card_swallowed",
                  "cash_withdrawal_charge",
                  "cash_withdrawal_not_recognised",
                  "change_pin",
                  "compromised_card",
                  "contactless_not_working",
                  "country_support",
                  "declined_card_payment",
                  "declined_cash_withdrawal",
                  "declined_transfer",
                  "direct_debit_payment_not_recognised",
                  "disposable_card_limits",
                  "edit_personal_details",
                  "exchange_charge",
                  "exchange_rate",
                  "exchange_via_app",
                  "extra_charge_on_statement",
                  "failed_transfer",
                  "fiat_currency_support",
                  "get_disposable_virtual_card",
                  "get_physical_card",
                  "getting_spare_card",
                  "getting_virtual_card",
                  "lost_or_stolen_card",
                  "lost_or_stolen_phone",
                  "order_physical_card",
                  "passcode_forgotten",
                  "pending_card_payment",
                  "pending_cash_withdrawal",
                  "pending_top_up",
                  "pending_transfer",
                  "pin_blocked",
                  "receiving_money",
                  "Refund_not_showing_up",
                  "request_refund",
                  "reverted_card_payment?",
                  "supported_cards_and_currencies",
                  "terminate_account",
                  "top_up_by_bank_transfer_charge",
                  "top_up_by_card_charge",
                  "top_up_by_cash_or_cheque",
                  "top_up_failed",
                  "top_up_limits",
                  "top_up_reverted",
                  "topping_up_by_card",
                  "transaction_charged_twice",
                  "transfer_fee_charged",
                  "transfer_into_account",
                  "transfer_not_received_by_recipient",
                  "transfer_timing",
                  "unable_to_verify_identity",
                  "verify_my_identity",
                  "verify_source_of_funds",
                  "verify_top_up",
                  "virtual_card_not_working",
                  "visa_or_mastercard",
                  "why_verify_identity",
                  "wrong_amount_of_cash_received",
                  "wrong_exchange_rate_for_cash_withdrawal"]
    index2label = dict(enumerate(class_list))
    label2index = dict([(v, k) for k, v in index2label.items()])

    def _collect(self, csv_file: str) -> Iterator[Tuple[int, str]]:
        with open(csv_file, encoding="utf-8") as f:
            data = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            # data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace = True)
            next(data)
            for line_id, line in enumerate(data):
                yield line_id, line

    def _cache_key_function(self, line_info: Tuple[int, str]) -> str:
        return str(line_info[0])

    def _parse_pack(self, line_info: Tuple[int, str]) -> Iterator[DataPack]:
        line_id, line = line_info
        # content label title
        pack = DataPack()
        request_text, label = line

        pack.pack_name = line_id
        text, indices = generate_text_n_anchor(["content"], request_text)
        pack.set_text(text, replace_func=self.text_replace_operation)
        class_id = self.label2index[label]
        doc = Document(pack, 0, indices["content"][1])
        doc.document_class = [self.configs.class_names[class_id]]
        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()
        config.update(
            {
                "candidate_labels": cls.index2label
            }
        )
        return config


class AGNewsReader(PackReader):
    r""":class:`AGNewsReader` is designed to read in AG News
    text classification dataset.
    The AG's news topic classification dataset is constructed by Xiang Zhang
    (xiang.zhang@nyu.edu) from the AG corpus. It is used as a text
    classification benchmark in the following paper:
    https://arxiv.org/abs/1509.01626
    The dataset can be downloaded from:
    https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

    The input to this reader is the path to the CSV file.
    """

    def _collect(  # type: ignore
            self, csv_file: str
    ) -> Iterator[Tuple[int, str]]:
        r"""Collects from a CSV file path and returns an iterator of AG News
        data. The elements in the iterator correspond to each line
        in the csv file. One line is expected to be parsed as one
        DataPack.

        Args:
            csv_file: A CSV file path.

        Returns: Iterator of each line in the csv file.
        """
        with open(csv_file, "r") as f:
            for line_id, line in enumerate(f):
                yield line_id, line

    def _cache_key_function(self, line_info: Tuple[int, str]) -> str:
        return str(line_info[0])

    def _parse_pack(self, line_info: Tuple[int, str]) -> Iterator[DataPack]:
        line_id, line = line_info

        pack = DataPack()
        text: str = ""
        line = line.strip()
        data = line.split(",")

        class_id: int = int(data[0].replace('"', ""))
        title: str = data[1]
        description: str = data[2]

        text += title
        title_end = len(text)
        text += "\n" + description
        description_start = title_end + 1
        description_end = len(text)

        pack.set_text(text, replace_func=self.text_replace_operation)

        doc = Document(pack, 0, description_end)
        doc.document_class = [self.configs.class_names[class_id]]
        Title(pack, 0, title_end)
        Description(pack, description_start, description_end)

        pack.pack_name = line_id
        yield pack

    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()

        config.update(
            {
                "candidate_labels": {
                    1: "World",
                    2: "Sports",
                    3: "Business",
                    4: "Sci/Tech",
                }
            }
        )
        return config
