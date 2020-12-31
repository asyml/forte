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
In this program we try to implement the famous Eliza chatbot, a rule-based
chatbot invented in 1964. The rule-based and model-less nature makes it
extremely suitable for demonstration purposes. Note that this implementation
may not reproduce the exact behavior of the original Eliza.

For more information of Eliza, see https://en.wikipedia.org/wiki/ELIZA and
the paper at: https://dl.acm.org/doi/10.1145/365153.365168
"""
import logging
from typing import Optional

from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance


class ElizaProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        utterance: Optional[Utterance] = None
        u: Utterance
        for u in input_pack.get(Utterance):
            if u.speaker == 'user':
                utterance = u

        if not utterance:
            logging.info("Cannot get new user utterance.")
