#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
from typing import Dict, List, Callable, Any

from torch import Tensor

from forte.data.extractor.extractor import BaseExtractor

# TODO: this class should be replaced with existing library like
#  pytorch lightning
class Trainer:
    def __init__(self,
                 create_model_fn: Callable,
                 create_optim_fn: Callable,
                 pass_tensor_to_model_fn: Callable):
        self.create_model_fn = create_model_fn
        self.create_optim_fn = create_optim_fn
        self.pass_tensor_to_model_fn = pass_tensor_to_model_fn
        self.model = None
        self.optim = None

    def setup(self, schemes: Dict[str, Dict[str, Any]]):
        self.model = self.create_model_fn(schemes)
        self.optim = self.create_optim_fn(self.model)

    def train(self, batch_tensors: Dict[str, Dict[str, Tensor]]):

        step = 0
        train_err = 0.0
        train_total = 0.0

        step += 1
        self.optim.zero_grad()

        # Pass a batch data to the model
        loss = self.pass_tensor_to_model_fn(self.model, batch_tensors)

        loss.backward()
        self.optim.step()

        num_inst = len(next(iter(next(iter(batch_tensors.values())))))
        train_err += loss.item() * num_inst
        train_total += num_inst
