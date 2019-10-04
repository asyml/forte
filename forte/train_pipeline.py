import logging
from typing import Optional, List, Iterator

from texar.torch import HParams

from forte.data import PackType
from forte.pipeline import Pipeline
from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.data.readers import BaseReader
from forte.processors.base import BaseProcessor
from forte.trainer.base import BaseTrainer

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self, train_reader: BaseReader, trainer: BaseTrainer,
                 dev_reader: BaseReader, configs: HParams,
                 preprocessors: Optional[List[BaseProcessor]] = None,
                 evaluator: Optional[Evaluator] = None,
                 predictor: Optional[BaseProcessor] = None):
        self.resource = Resources()
        self.configs = configs

        train_reader.initialize(self.resource, self.configs)

        if preprocessors is not None:
            for p in preprocessors:
                p.initialize(resource=self.resource,
                             configs=configs.preprocessor)
            self.preprocessors = preprocessors
        else:
            self.preprocessors = []

        """trainer.initialize(self.resource, configs)

        if predictor is not None:
            logger.info(
                "Training pipeline initialized with real eval setting."
            )
            predictor_config = configs.predictor
            predictor.initialize(self.resource, predictor_config)"""

        self.train_reader = train_reader
        self.trainer = trainer
        self.predictor = predictor
        self.evaluator = evaluator
        self.dev_reader = dev_reader

    def run(self):
        logging.info("The pipeline is running preparation.")
        self.prepare()
        self.trainer.initialize(self.resource, self.configs)
        if self.predictor is not None:
            logger.info("Training pipeline initialized with real eval setting.")
            # predictor_config = self.configs.predictor
            self.predictor.initialize(self.resource, self.configs)

        logging.info("The pipeline is training")
        self.train()
        # self.eval_dev(epoch=0)

    def prepare(self, *args, **kwargs) -> Iterator[PackType]:
        prepare_pl = Pipeline()
        prepare_pl.set_reader(self.train_reader)
        for p in self.preprocessors:
            prepare_pl.add_processor(p)

        prepare_pl.run(conll_directory=self.configs.config_data.val_path)

        for p in self.preprocessors:
            p.finish(resource=self.resource)

    def train(self):
        pack_count = 0
        epoch = 0
        while True:
            epoch += 1
            # we need to have directory ready here
            # for pack in self.prepare(data_source=self.configs.train_path):
            for pack in self.train_reader.iter(conll_directory=self.configs.config_data.train_path):
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
                    conll_directory=self.configs.config_data.val_path):
                predicted_pack = pack.view()
                self.predictor.process(predicted_pack)
                self.evaluator.consume_next(pack, predicted_pack)
            validation_result["eval"] = self.evaluator.get_result()

            """for pack in self.dev_reader.iter(
                    conll_directory=self.configs.config_data.test_path):
                predicted_pack = pack.view()
                self.predictor.process(predicted_pack)
                self.evaluator.consume_next(pack, predicted_pack)
            validation_result["test"] = self.evaluator.get_result()"""

        return validation_result

    def finish(self):
        self.train_reader.finish(self.resource)
        self.dev_reader.finish(self.resource)
        for p in self.preprocessors:
            p.finish(self.resource)
        self.predictor.finish(self.resource)
