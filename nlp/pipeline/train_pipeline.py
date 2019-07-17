import logging
from typing import Optional

from nlp.pipeline.common.evaluation import Evaluator
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data.readers.base_reader import BaseReader
from nlp.pipeline.processors import BaseProcessor
from nlp.pipeline.trainer.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(
            self,
            train_reader: BaseReader,
            trainer: BaseTrainer,
            dev_reader: BaseReader,
            # TODO: Let's define the config system.
            # config,
            resource: Optional[Resources] = None,
            evaluator: Optional[Evaluator] = None,
            predictor: Optional[BaseProcessor] = None,
    ):
        resource.save()
        # resource = Resources(config)
        trainer.initialize(resource)

        if predictor is not None:
            logger.info(
                "Training pipeline initialized with real eval setting."
            )
            predictor.initialize(resource)
            predictor.set_mode(overwrite=False)

        self.train_reader = train_reader
        self.trainer = trainer
        self.predictor = predictor
        self.evaluator = evaluator
        self.dev_reader = dev_reader
        self.config_data = resource.resources["config_data"]

    def train(self):
        pack_count = 0
        epoch = 0
        while True:
            epoch += 1
            # we need to have directory ready here
            for pack in self.train_reader.dataset_iterator(
                    self.config_data.train_path
            ):
                # data_request is a string. How to transform it to the
                # function parameters? Or we can change the interface of
                # get_data
                # What if we want to do validate after several steps? We
                # need to set this in the trainer.
                for instance in pack.get_data(**self.trainer.data_request()):
                    if self.trainer.validation_requested():
                        dev_res = self.eval_dev(epoch)
                        self.trainer.validation_done()
                        self.trainer.post_validation_action(dev_res)
                    if self.trainer.stop_train():
                        return

                    # TODO: Change to consume
                    self.trainer.process(instance)
                self.trainer.pack_finish_action(pack_count)
            self.trainer.epoch_finish_action(epoch)
            # Cannot call the `trainer.finish` function explicitly here since
            # there is a return

    def eval_dev(self, epoch: int):

        validation_result = {"epoch": epoch}

        if self.predictor is not None and self.evaluator is not None:
            for pack in self.dev_reader.dataset_iterator(
                    self.config_data.val_path
            ):
                self.predictor.process(pack)
                self.evaluator.consume_next(pack)
            validation_result["eval"] = self.evaluator.get_result()

            for pack in self.dev_reader.dataset_iterator(
                    self.config_data.test_path
            ):
                self.predictor.process(pack)
                self.evaluator.consume_next(pack)
            validation_result["test"] = self.evaluator.get_result()

        return validation_result
