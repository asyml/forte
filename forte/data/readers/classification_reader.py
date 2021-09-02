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
from typing import Iterator, Tuple

from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.ag_news import Description
from ft.onto.base_ontology import *

__all__ = [
    "AGNewsReader",
    "AmazonPolarityReader",
    "Banking77"
]


# PackReader:

# _collect:  file_path ->  takes input and yields data (actually returns an iterator)
            # line_id, line
# _parse_pack: another function returns an iterator
            # parse the data pack which is what _collect returns (line_id, line)
            # class DataPack to collect line info
# _cache_key_function:
            #
# classmethod: default_configs
            # stores class id with corresponidng class names


def generate_text_n_anchor(subtext_names, *args):
    """
    A generic method to generate start and end indices for every subtext parameters, such as titles, content
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

    """

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
        label, title, content = line

        pack.pack_name = line_id
        text, indices = generate_text_n_anchor(["title", "content"], title, content)
        pack.set_text(text, replace_func=self.text_replace_operation)
        class_id = int(label) - 1


        doc = Document(pack, 0, indices["content"][1])
        doc.document_class = [self.configs.class_names[class_id]]
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
                "class_names": {
                    0: "negative",
                    1: "positive"

                }
            }
        )
        return config


class Banking77(PackReader):
    r"""

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

        sent = Sentence(pack, 0, indices["content"][1])

        sent.classification = [self.configs.class_names[class_id]]
        # Title(pack, 0, indices["title"][1])
        # TODO: maybe another generic name for Description (mainSubText)
        # Description(pack, indices["content"][0],  indices["content"][1])

        pack.pack_name = line_id
        yield pack


    @classmethod
    def default_configs(cls):
        config: dict = super().default_configs()


        config.update(
            {
                "class_names": cls.index2label
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
                "class_names": {
                    1: "World",
                    2: "Sports",
                    3: "Business",
                    4: "Sci/Tech",
                }
            }
        )
        return config


