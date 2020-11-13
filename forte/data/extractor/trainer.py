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
import logging
from typing import Dict, List, Callable, Any, Optional

from texar.torch.data import Batch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from forte.data.extractor.extractor import BaseExtractor

logger = logging.getLogger(__name__)


# TODO: this class should be replaced with existing library like
#  pytorch lightning
class Trainer:
    def __init__(self,
                 create_model_fn: Callable,
                 create_optim_fn: Callable,
                 create_criterion_fn: Callable,
                 train_forward_fn: Callable):
        self.create_model_fn = create_model_fn
        self.create_optim_fn = create_optim_fn
        self.create_criterion_fn = create_criterion_fn
        self.train_forward_fn = train_forward_fn
        self.model: Optional[nn.Module] = None
        self.optim: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None

    def setup(self, schemes: Dict[str, Dict[str, Any]]):
        self.model = self.create_model_fn(schemes)
        self.optim = self.create_optim_fn(self.model)
        self.criterion = self.create_criterion_fn(self.model)

    def train(self, batch: Batch):
        self.optim.zero_grad()

        # Pass a batch data to the model
        loss = self.train_forward_fn(self.model, self.criterion, batch)

        loss.backward()
        self.optim.step()

        train_err = loss.item() * batch.batch_size

        return train_err, batch.batch_size
