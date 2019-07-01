from nlp.pipeline.trainer import Trainer
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.processors.predictor import Predictor
from nlp.pipeline.common.evaluation import Evaluator
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.base_reader import BaseReader
import logging


class TrainPipeline:
    def __init__(self,
                 train_reader: BaseReader,
                 trainer: Trainer,
                 dev_reader: BaseReader,
                 # TODO: Let's define the config system.
                 config,
                 evaluator: Evaluator = None,
                 predictor: Predictor = None,
                 ):
        resource = Resources(**config)
        trainer.initialize(resource)

        if predictor is not None:
            logging.info(
                "Training pipeline initialized with real eval setting."
            )
            predictor.initialize(resource)

        self.train_reader = train_reader
        self.trainer = trainer
        self.predictor = predictor
        self.evaluator = evaluator
        self.dev_reader = dev_reader

    def train(self):
        epoch = 0
        pack_count = 0
        while True:
            for pack in self.train_reader.dataset_iterator():
                for instance in pack.get_data(self.trainer.data_request()):
                    if self.trainer.validation_requested():
                        self.trainer.eval_call_back(self.eval_dev())
                    if self.trainer.stop_train():
                        return
                    # collect a batch of instances
                    self.trainer.process(instance)
                self.trainer.pack_finish_action(pack_count)
            self.trainer.epoch_finish_action(epoch)

    def eval_dev(self):
        def dev_instances():
            for dev_pack in self.dev_reader.dataset_iterator():
                for instance in dev_pack.get_data(self.trainer.data_request()):
                    yield instance

        validation_result = {
            "loss": self.trainer.get_loss(dev_instances())
        }

        if self.predictor is not None and self.evaluator is not None:
            for pack in self.dev_reader.dataset_iterator():
                self.predictor.process(pack)
                self.evaluator.consume_next(pack)
            validation_result['eval'] = self.evaluator.get_result()

        return validation_result
