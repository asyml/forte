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
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance, UtteranceContext


class Model:
    """This is a dummy model that always return the same string."""

    def __init__(self):
        self.model_str = "this content is from the model."

    def response(self):
        # Pretend the model is running for 5 seconds.
        print('model is computing...')
        import time
        time.sleep(5)
        return self.model_str


class ContentRewriter(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init

        # Make sure the initialize model here.
        self.model = Model()

    def new_utternace(self, input_pack: DataPack, text: str, speaker: str):
        input_pack.set_text(input_pack.text + '\n' + text)
        # And then mark this as a new utterance.
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

        self.new_utternace(input_pack, self.model.response(), 'ai')
