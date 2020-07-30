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

from examples.Cliner.CliNER.code.train import CliNERTrain


class CliTrain():
    def __init__(self):
        # Setup model path.
        self.txt = os.path.join(
            'CliNER/data/train_data/*.txt')
        self.con = os.path.join(
            'CliNER/data/train_data/*.con')
        self.output = os.path.join(
            'CliNER/data/test_predictions')
        self.model_path = os.path.join(
            'CliNER/models/train_full.model')
        self.format = 'i2b2'
        # pylint: disable=attribute-defined-outside-init
        self.model = CliNERTrain(self.txt, self.con, self.output,
                                 self.model_path, self.format)

    def train(self):
        self.model.train()
