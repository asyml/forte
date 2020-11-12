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
import unittest

import torch
from texar.torch.data import Batch
from torch import nn, Tensor
from typing import Dict, Any, List, Optional

from torch.optim import SGD

from forte.data.extractor.trainer import Trainer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(2, 3)
        self.loss_hooked: Tensor = torch.tensor([])

    def forward(self, input1, input2, input1_mask, input2_mask):
        # Do some fake calculation
        output1 = self.linear1((input1 * input1_mask).float())
        output2 = self.linear2((input2 * input2_mask).float())
        output2 = torch.sum(output2, dim=-1)

        self.loss_hooked: Tensor = torch.sum(output1 + output2)

        return self.loss_hooked


class TrainPipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "nesterov": True
        }

        self.model: MockModel = MockModel()

        def create_model_fn(schemes: Dict[str, Dict[str, Any]]):
            self.assertTrue("input1" in schemes)
            self.assertTrue("extractor" in schemes["input1"])
            self.assertTrue("converter" in schemes["input1"])
            self.assertTrue("input2" in schemes)
            self.assertTrue("extractor" in schemes["input2"])
            self.assertTrue("converter" in schemes["input2"])

            return self.model

        def create_optim_fn(model):
            optim = SGD(
                model.parameters(),
                lr=self.config["learning_rate"],
                momentum=self.config["momentum"],
                nesterov=self.config["nesterov"])
            return optim

        def create_criterion_fn(model: nn.Module) -> Optional[nn.Module]:
            return None

        def train_forward_fn(model, criterion: Optional[nn.Module],
                             tensors: Dict[str, Dict[str, Any]]):
            input1 = tensors["input1"]["tensor"]
            input1_mask = tensors["input1"]["mask"][0]
            input2 = tensors["input2"]["tensor"]
            input2_mask = tensors["input2"]["mask"][1]

            loss = model(input1, input2, input1_mask, input2_mask)

            return loss

        self.create_model_fn = create_model_fn
        self.create_optim_fn = create_optim_fn
        self.create_criterion_fn = create_criterion_fn
        self.train_forward_fn = train_forward_fn

        self.trainer = Trainer(create_model_fn=create_model_fn,
                               create_optim_fn=create_optim_fn,
                               create_criterion_fn=create_criterion_fn,
                               train_forward_fn=train_forward_fn)

    def test_train(self):
        self.trainer.setup({"input1": {
            "extractor": None,
            "converter": None
        }, "input2": {
            "extractor": None,
            "converter": None
        }})

        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optim)

        input1: Tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        input1_mask: List[Tensor] = [
            torch.tensor([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
        ]

        input2: Tensor = torch.tensor([[[2, 3], [1, 2], [4, 5]],
                                       [[1, 0], [5, 6], [3, 4]],
                                       [[1, 9], [5, 6], [1, 0]]])
        input2_mask: List[Tensor] = [
            torch.tensor([[1, 0, 0],
                          [1, 1, 0],
                          [1, 1, 1]]),
            torch.tensor([[[1, 0], [1, 1], [0, 0]],
                          [[1, 1], [1, 0], [1, 1]],
                          [[1, 0], [1, 0], [1, 0]]])
        ]

        batch_tensors: Dict[str, Dict[str, Tensor]] = {
            "input1": {
                "tensor": input1,
                "mask": input1_mask
            },
            "input2": {
                "tensor": input2,
                "mask": input2_mask
            }
        }

        batch = Batch(3, **batch_tensors)

        gold_model: MockModel = MockModel()

        gold_model.linear1.weight.data = \
            self.trainer.model.linear1.weight.data.clone()
        gold_model.linear1.bias.data = \
            self.trainer.model.linear1.bias.data.clone()
        gold_model.linear2.weight.data = \
            self.trainer.model.linear2.weight.data.clone()
        gold_model.linear2.bias.data = \
            self.trainer.model.linear2.bias.data.clone()

        self.expected_loss = gold_model.forward(input1,
                                                input2,
                                                input1_mask[0],
                                                input2_mask[1])

        self.trainer.train(batch)

        self.actual_loss = self.model.loss_hooked

        self.assertTrue(torch.allclose(self.actual_loss,
                                       self.expected_loss))


if __name__ == '__main__':
    unittest.main()
