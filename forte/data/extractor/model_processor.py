#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from abc import abstractmethod
from typing import Dict
from texar.torch.data import Batch

from forte.common.configuration import Config
from forte.data.base_pack import PackType
from forte.data.extractor.extractor import BaseExtractor


class ModelProcessor():
    def __init__(self):
        pass

    @abstractmethod
    def setup(self,
              schemes: Dict[str, Dict[str, BaseExtractor]],
              pl_config: Config):
        raise NotImplementedError

    @abstractmethod
    def train(self, batch: Batch):
        raise NotImplementedError

    @abstractmethod
    def train_finish(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, pred_pack: PackType, ref_pack: PackType):
        raise NotImplementedError

    @abstractmethod
    def evaluate_finish(self, epoch: int):
        raise NotImplementedError
