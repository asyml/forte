from abc import abstractmethod
from typing import Dict, Iterator

from nlp.pipeline.common.resources import Resources
from nlp.pipeline.processors.base_processor import BaseProcessor


class Trainer(BaseProcessor):
    def __init__(self, config):
        super().__init__()
        self._stop_train = False
        self._validation_requested = False
        self._dev_eval_result = None

    @abstractmethod
    def initialize(self, resources: Resources):
        raise NotImplemented

    def validation_requested(self) -> bool:
        return self._validation_requested

    def stop_train(self) -> bool:
        return self._stop_train

    @abstractmethod
    def data_request(self):
        pass

    @abstractmethod
    def process(self, instance: Dict):
        # Do training
        raise NotImplementedError

    def get_loss(self, instances: Iterator[Dict]):
        raise NotImplementedError

    def pack_finish_action(self, pack_count: int):
        pass

    def epoch_finish_action(self, epoch_num: int):
        pass

    def request_eval(self):
        self._validation_requested = True

    def request_stop_train(self):
        self._stop_train = True

    @abstractmethod
    def eval_call_back(self, eval_result):
        pass
        # self.__dev_eval_result = eval_result
        # self.__validation_requested = False
