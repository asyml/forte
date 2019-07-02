from nlp.pipeline.trainer.trainer import Trainer
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
                 # # TODO: Let's define the config system.
                 # config,
                 resource: Resources = None,
                 evaluator: Evaluator = None,
                 predictor: Predictor = None,
                 ):
        # resource = Resources(config)
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
        self.config_data = resource.resources['config_data']

    def train(self):
        pack_count = 0
        epoch = 0
        while True:
            epoch += 1
            # we need to have directory ready here
            for pack in self.train_reader.dataset_iterator(
                    self.config_data.train_path):
                # data_request is a string. How to transform it to the
                # function parameters? Or we can change the interface of
                # get_data
                # What if we want to do validate after several steps? We
                # need to set this in the trainer.
                for instance in pack.get_data(**self.trainer.data_request()):
                    if self.trainer.validation_requested():
                        self.trainer.eval_call_back(self.eval_dev())
                    if self.trainer.stop_train():
                        return
                    self.trainer.process(instance)
                    print(f'request_Eval? {self.trainer.validation_requested()}')
                self.trainer.pack_finish_action(pack_count)
            self.trainer.epoch_finish_action(epoch)

    def eval_dev(self):

        # define pack first
        # def dev_instances():
        #     for instance in pack.get_data(self.trainer.data_request()):
        #         yield instance

        # validation_result = {
        #     "loss": self.trainer.get_loss(dev_instances())
        # }
        validation_result = {}

        if self.predictor is not None and self.evaluator is not None:
            for pack in self.dev_reader.dataset_iterator(
                    self.config_data.val_path):
                self.predictor.process(pack)
                print('add_cnt:{}'.format(self.predictor.add_cnt))
                self.evaluator.consume_next(pack)
            validation_result['eval'] = self.evaluator.get_result()

            for pack in self.dev_reader.dataset_iterator(
                    self.config_data.test_path):
                self.predictor.process(pack)
                self.evaluator.consume_next(pack)
            validation_result['test'] = self.evaluator.get_result()

        return validation_result

