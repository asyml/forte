from abc import abstractmethod
from typing import Dict, Iterator

from nlp.pipeline.common.resources import Resources
from nlp.pipeline.processors.base_processor import BaseProcessor


class BaseTrainer(BaseProcessor):
    def __init__(self, config):  # pylint: disable=unused-argument
        super().__init__()
        self._stop_train = False
        self._validation_requested = False

    @abstractmethod
    def initialize(self, resources: Resources):
        """
        The training pipeline will run this initialization method during
        the initialization phase and send resources in as parameters.
        Args:
            resources: The resources required for training.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def data_request(self):
        pass

    @abstractmethod
    def process(self, instance: Dict):
        # Do training
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
