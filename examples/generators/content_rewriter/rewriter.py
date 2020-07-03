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
from typing import Dict, Any

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance, UtteranceContext
from examples.generators.content_rewriter. \
    model.manip import Rewriter
from examples.generators.content_rewriter.model import utils_e2e_clean
from examples.generators.content_rewriter.model import config_data_e2e_clean
from examples.generators.content_rewriter.model import manip


class ContentRewriter(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        # Setup model path.
        utils_e2e_clean.load_e2e_ents(
            os.path.join(configs.model_dir, 'e2e_data', 'x_value.vocab.txt'))
        config_data_e2e_clean.dataset_dir = os.path.join(
            configs.model_dir, 'e2e_data')
        config_data_e2e_clean.set_datas()
        manip.config_data.dataset_dir = os.path.join(
            configs.model_dir, 'e2e_data')
        manip.expr_name = os.path.join(configs.model_dir, "e2e_model", "demo")

        manip.set_model_dir(manip.expr_name)

        # pylint: disable=attribute-defined-outside-init
        self.model = Rewriter()
        self.model.load_model()

    def new_utternace(self, input_pack: DataPack, text: str, speaker: str):
        # os.system('./forte.sh')
        print(text)
        input_pack.set_text(input_pack.text + '\n' + text)
        # And then mark this as a new utterance.
        print('The response is:')
        print(text)

        u = Utterance(input_pack,
                      len(input_pack.text) - len(text),
                      len(input_pack.text))
        u.speaker = speaker

    def _process(self, input_pack: DataPack):
        context = input_pack.get_single(UtteranceContext)

        # Make sure we take the last utterance.
        utterance: Utterance
        for u in input_pack.get(Utterance):
            utterance = u

        print('The input context is:')
        print(context.text)

        print('The utterance is:')
        print(utterance.text)

        self.new_utternace(input_pack, self.model.eval_epoch('test'), 'ai')

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config['model_dir'] = 'content_rewriter/model'
