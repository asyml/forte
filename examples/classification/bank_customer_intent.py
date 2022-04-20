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
import sys
from importlib_metadata import csv
from termcolor import colored

from forte import Pipeline
from forte.data.readers import ClassificationDatasetReader
from fortex.nltk import NLTKSentenceSegmenter
from fortex.huggingface import ZeroShotClassifier
from ft.onto.base_ontology import Sentence


csv_path = "data_samples/banking77/sample.csv"
pl = Pipeline()
# initialize labels
class_names = [
    "activate_my_card",
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
    "wrong_exchange_rate_for_cash_withdrawal",
]
index2class = dict(enumerate(class_names))

# initialize reader config
this_reader_config = {
    "forte_data_fields": [
        "ft.onto.base_ontology.Body",
        "label",
    ],
    "index2class": index2class,
    "text_fields": ["ft.onto.base_ontology.Body"],
    "digit_label": False,
    "one_based_index_label": False,
}

pl.set_reader(ClassificationDatasetReader(), config=this_reader_config)
pl.add(NLTKSentenceSegmenter())
pl.add(ZeroShotClassifier(), config={"candidate_labels": class_names})
pl.initialize()

for pack in pl.process_dataset(csv_path):
    for sentence in pack.get(Sentence):
        if (
            input("Type n for the next sentence and its prediction: ").lower()
            == "n"
        ):
            sent_text = sentence.text
            print(colored("Sentence:", "red"), sent_text, "\n")
            print(colored("Prediction:", "blue"), sentence.classification)
        else:
            print("Exit the program due to unrecognized input")
            sys.exit()
