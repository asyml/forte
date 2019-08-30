import logging
from typing import Optional

from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.data.readers import BaseReader
from forte.processors.base import BaseProcessor
from forte.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(
            self,
            train_reader: BaseReader,
            trainer: BaseTrainer,
            dev_reader: BaseReader,
            resource: Resources,
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
            predictor.initialize(configs=None, resource=resource)

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
            for pack in self.train_reader.iter(
                    data_source=self.config_data.train_path
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

                    self.trainer.consume(instance)
                self.trainer.pack_finish_action(pack_count)
            self.trainer.epoch_finish_action(epoch)
            # Cannot call the `trainer.finish` function explicitly here since
            # there is a return

    def eval_dev(self, epoch: int):

        validation_result = {"epoch": epoch}

        if self.predictor is not None and self.evaluator is not None:
            for pack in self.dev_reader.iter(
                    data_source=self.config_data.val_path
            ):
                predicted_pack = pack.view()
                self.predictor.process(predicted_pack)
                self.evaluator.consume_next(pack, predicted_pack)
            validation_result["eval"] = self.evaluator.get_result()

            for pack in self.dev_reader.iter(
                    data_source=self.config_data.test_path
            ):
                predicted_pack = pack.view()
                self.predictor.process(predicted_pack)
                self.evaluator.consume_next(pack, predicted_pack)
            validation_result["test"] = self.evaluator.get_result()

        return validation_result
