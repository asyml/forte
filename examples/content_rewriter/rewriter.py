# Copyright 2019 The Forte Authors. All Rights Reserved.
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
The re-writer processor
"""
import os
from typing import Dict, Any, Optional
import logging

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance, UtteranceContext
from model.manip import Rewriter
from model import utils_e2e_clean
from model import config_data_e2e_clean
from model import manip


class ContentRewriter(PackProcessor):
    def initialize(self, _: Resources, configs: Config):
        # Setup model path.
        utils_e2e_clean.load_e2e_ents(
            os.path.join(configs.model_dir, "e2e_data", "x_value.vocab.txt")
        )
        config_data_e2e_clean.dataset_dir = os.path.join(
            configs.model_dir, "e2e_data"
        )
        config_data_e2e_clean.set_datas()

        manip.Config.set_path(
            os.path.join(configs.model_dir, "e2e_model", "demo")
        )

        # pylint: disable=attribute-defined-outside-init
        self.model = Rewriter()
        self.model.load_model()

    def new_utternace(self, input_pack: DataPack, text: str, speaker: str):
        input_pack.set_text(input_pack.text + "\n" + text)
        logging.info("The response is:")
        logging.info(text)

        print("The response is:")
        print(text)

        u = Utterance(
            input_pack, len(input_pack.text) - len(text), len(input_pack.text)
        )
        u.speaker = speaker

    def _process(self, input_pack: DataPack):
        context: UtteranceContext = input_pack.get_single(  # type: ignore
            UtteranceContext
        )

        # Make sure we take the last utterance from the user.
        utterance: Optional[Utterance] = None
        u: Utterance
        for u in input_pack.get(Utterance):
            if u.speaker == "user":
                utterance = u

        if utterance:
            logging.info("The content is %s", context.text)
            logging.info("The sample utterance is %s", utterance)

            self.prepare_data(context, utterance)
            self.new_utternace(input_pack, self.model.eval_epoch("test"), "ai")
        else:
            logging.info("Cannot get another utterance.")
            self.new_utternace(
                input_pack,
                "Hey, I didn't get what you say, could you try again?",
                "ai",
            )

    def prepare_data(self, context: UtteranceContext, utterance: Utterance):
        logging.info("Preparing test data with the context and utterance")
        logging.info("Context is : %s", context.text)
        logging.info("Utterance is : %s", utterance.text)

        type = []
        val = []
        asso = []

        for triple in context.text.split():
            for idx, i in enumerate(triple.split("|")):
                if not idx:
                    val.append(i)
                elif idx == 1:
                    type.append(i)
                else:
                    asso.append(i)
        data_dir = os.path.join(config_data_e2e_clean.dataset_dir, "test")
        logging.info("Writing to data dir: %s", data_dir)

        with open("{}/x_type.test.txt".format(data_dir), "w") as f_type, open(
            "{}/x_value.test.txt".format(data_dir), "w"
        ) as f_val, open(
            "{}/x_associated.test.txt".format(data_dir), "w"
        ) as f_asso, open(
            "{}/y_ref.test.txt".format(data_dir), "w"
        ) as f_ref:
            f_type.write(" ".join(type))
            f_val.write(" ".join(val))
            f_asso.write(" ".join(asso))
            f_ref.write(utterance.text)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "model_dir": "content_rewriter/model"
        }
