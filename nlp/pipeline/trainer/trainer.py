from abc import abstractmethod
from typing import Dict, Any, Iterator
from nlp.pipeline.common.evaluation import Evaluator
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.common.resources import Resources


class Trainer(BaseProcessor):
    def __init__(self, config):
        super().__init__()
        self.__stop_train = False
        self.__validation_requested = False
        self.__dev_eval_result = None

    def initialize(self, resources: Resources):
        pass

    def validation_requested(self) -> bool:
        return self.__validation_requested

    def stop_train(self) -> bool:
        return self.__stop_train

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
        self.__validation_requested = True

    def _eval_call_back(self, eval_result):
        self.__dev_eval_result = eval_result
        self.__validation_requested = False
