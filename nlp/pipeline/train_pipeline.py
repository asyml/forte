from nlp.pipeline.trainer import Trainer
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.processors.predictor import Predictor
from nlp.pipeline.common.evaluation import Evaluator
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.base_reader import BaseReader


class TrainPipeline:
    def __init__(self,
                 train_reader: BaseReader,
                 trainer: Trainer,
                 evaluator: Evaluator,
                 # TODO: Let's define the config system.
                 config,
                 predictor: Predictor = None,
                 dev_reader: BaseReader = None,
                 ):
        resource = Resources(config)
        trainer.initialize(resource)
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
                    if self.trainer.eval_requested():
                        self.trainer._eval_call_back(self.eval_dev())
                    if self.trainer.stop_train():
                        return
                    self.trainer.process(instance)
                self.trainer.pack_finish_action(pack_count)
            self.trainer.epoch_finish_action(epoch)

    def eval_dev(self):
        for pack in self.dev_reader.dataset_iterator():
            self.predictor.process(pack)
            self.evaluator.consume_next(pack)
        return self.evaluator.get_result()
