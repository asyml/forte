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

from abc import abstractmethod
from typing import Dict, Iterator

from texar.torch import HParams

from forte.common.resources import Resources
from forte.pipeline_component import PipelineComponent


class BaseTrainer(PipelineComponent):
    def __init__(self):  # pylint: disable=unused-argument
        super().__init__()
        self._stop_train = False
        self._validation_requested = False

    @abstractmethod
    def initialize(self, resource: Resources, configs: HParams):
        """
        The training pipeline will run this initialization method during
        the initialization phase and send resources in as parameters.
        Args:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def data_request(self):
        pass

    @abstractmethod
    def consume(self, instance: Dict):
        # consume the instance
        raise NotImplementedError

    @abstractmethod
    def post_validation_action(self, dev_res):
        """
        This method
        Returns:

        """
        pass

    @abstractmethod
    def get_loss(self, instances: Iterator[Dict]):
        raise NotImplementedError

    def update_resource(self):
        r"""Update the resource after every epoch which can be consumed by the
        predictor
        """
        raise NotImplementedError

    def pack_finish_action(self, pack_count: int):
        """
        This function will be called by the pipeline when one pack is
        finished.

        Args:
            pack_count:

        Returns:

        """
        pass

    def epoch_finish_action(self, epoch_num: int):
        """
        This function will be called by the pipeline when one epoch is
        finished. For example, the trainer can call request_stop_train()
        when the number of epoch reaches a predefined value.
        Args:
            epoch_num:

        Returns:

        """
        pass

    def request_eval(self):
        """
        The trainer should call this method to inform the pipeline to
        conduct evaluation.
        Returns:

        """
        self._validation_requested = True

    def request_stop_train(self):
        """
        The trainer should call this method to inform the pipeline to stop
        training.
        Returns:

        """
        self._stop_train = True

    def validation_done(self):
        """
        Used only by the pipeline to close the validation request.

        Returns:

        """
        self._validation_requested = False

    def validation_requested(self) -> bool:
        """
        Used only by the pipeline to check whether the trainer has made
        the validation request.

        Returns: True if the validation request is submitted and not completed.
        """
        return self._validation_requested

    def stop_train(self) -> bool:
        """
        Used only by the pipeline to check if the trainer decided to stop
        training.

        Returns: True if the trainer decided to stop.
        """
        return self._stop_train
