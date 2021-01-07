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
import logging
from abc import abstractmethod
from typing import Dict, Iterator, Any, Optional
import pickle

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.train_preprocessor import TrainPreprocessor

logger = logging.getLogger(__name__)

__all__ = [
    "BaseTrainer"
]


class BaseTrainer:
    def __init__(self):
        self.tp_request: Optional[Dict] = None
        self.tp_config: Optional[Dict] = None
        self.pack_generator: Optional[Iterator[DataPack]] = None
        self._tp: Optional[TrainPreprocessor] = None
        self._initialized: bool = False

    def initialize(self):
        # Check if initialize has already been called before
        if self._initialized:
            return

        self.tp_request: Dict = self.create_tp_request()
        self.tp_config: Dict = self.create_tp_config()
        self.pack_generator: Iterator[DataPack] = self.create_pack_generator()
        self._tp: TrainPreprocessor = \
            TrainPreprocessor(pack_generator=self.pack_generator,
                              request=self.tp_request,
                              config=self.tp_config)
        self._initialized = True

    @property
    def train_preprocessor(self) -> TrainPreprocessor:
        if not self._initialized:
            raise ValueError("initialize should be called to "
                             "build train preprocessor.")
        return self._tp

    def run(self):
        self.initialize()
        self.train()

    @abstractmethod
    def create_tp_request(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def create_tp_config(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def create_pack_generator(self) -> Iterator[DataPack]:
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def save(self, *args: Any, **kwargs: Any):
        # Check arg type. Default behavior only supports str as args[0] which
        # is considered as a disk file path.
        if type(args[0]) != str:
            raise ValueError("Do not support input args: {} and kwargs: {}"
                             .format(args, kwargs))

        file_path = args[0]
        request: Dict = self.train_preprocessor.request

        with open(file_path, "wb") as f:
            pickle.dump(request, f)
