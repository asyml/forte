#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
import logging

from forte.data.extractor.predictor import Predictor
from forte.pipeline import Pipeline
from texar.torch.data import DataIterator
from tqdm import tqdm
import torch
from torch import device
from typing import Optional, Dict, Type, Any, Union, Callable

from forte.data.extractor.unpadder import BaseUnpadder, SameLengthUnpadder
from forte.data.types import DATA_OUTPUT
from forte.data.extractor.data_pack_loader import DataPackLoader
from forte.data.extractor.data_pack_dataset import DataPackDataSource, \
    DataPackDataset
from forte.processors.base.base_processor import BaseProcessor
from forte.data.extractor.converter import Converter
from forte.common.configuration import Config
from forte.data.extractor.trainer import Trainer
from forte.data.ontology.core import EntryType
from forte.data.extractor.extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.evaluation.base import Evaluator
from forte.data.readers.base_reader import PackReader

logger = logging.getLogger(__name__)


# TODO: BasePack or DataPack

class TrainPipeline:
    """
    TrainPipeline serves as the manager of training a model including: build
    vocabulary, preprocess and extract the features, do the actual training and
    evaluation. The training entry point is `run` method.

    `TrainPipeline` will maintain a Config that stores all the configurable
    parameters for various components inside TrainPipeline. In addition, those
    parameters are organized in a hierarchical way.

    For example, at first level, it will contain four parts: pipeline,
    train, data_pack and dataset. The `default_config` will return the
    default value for all configurable parameters.

    Particularly, the `dataset` config is exactly the same as what
    texar.torch.data.DatasetBase::default_config will return.

    `TrainPipeline` will also accept a user request given as the input to the
    method `run`. Internally it will parse this user request and store the
    parsed result as a dictionary into `feature_resource`.
    An example `feature_resource` is:
    feature_resource: {
        "scope": ft.onto.Sentence
        "schemes": {
            "text_tag": {
                "extractor":  Extractor,
                "converter": Converter,
                "type": DATA_INPUT
            },
            "char_tag" {
                "extractor":  Extractor,
                "converter": Converter,
                "type": DATA_INPUT
            }
            "ner_tag": {
                "extractor":  Extractor,
                "converter": Converter,
                "unpadder": Unpadder,
                "type": DATA_OUTPUT
            }
        }
    }
    """

    def __init__(self,
                 train_reader: PackReader,
                 val_reader: PackReader,
                 trainer: Trainer,
                 predict_forward_fn: Callable,
                 evaluator: Optional[Evaluator] = None,
                 config: Optional[Union[Config, Dict]] = None):
        self._config = Config(config, default_hparams=self.default_configs())
        self._validate_config()

        self._train_reader = train_reader
        self._val_reader = val_reader

        self._predict_forward_fn = predict_forward_fn
        self._trainer: Trainer = trainer

        if self._config.pipeline.evaluate:
            assert evaluator is not None, \
                "Evaluator should be set when evaluate is enabled."
            self._predictor: Optional[Predictor] = None
            self._evaluator: Optional[Evaluator] = evaluator

        self._train_data_pack_loader = \
            DataPackLoader(reader=self._train_reader,
                           cache_dir=self._config.data_pack.train_cache_dir,
                           config=self._config.data_pack.train_loader)

        self._val_data_pack_loader = \
            DataPackLoader(reader=self._val_reader,
                           cache_dir=self._config.data_pack.val_cache_dir,
                           config=self._config.data_pack.val_loader)

        self._user_request: Dict = {}
        self._feature_resource: Dict = {}

        if self._config.pipeline.logging:
            logging.basicConfig(level=logging.INFO)

    @staticmethod
    def default_configs():
        return {
            "pipeline": {
                "evaluate": True,
                "logging": True
            },
            "data_pack": {
                "train_loader": DataPackLoader.default_configs(),
                "train_cache_dir": ".train_data_pack_cache",
                "val_loader": DataPackLoader.default_configs(),
                "val_cache_dir": ".val_data_pack_cache"
            },
            "train": {
                "device": torch.device("cpu"),
                "num_epochs": 10,
            },
            "dataset": DataPackDataset.default_hparams()
        }

    def _validate_config(self):
        # Placeholder
        pass

    def _parse_request(self, data_request: Dict):
        """
        Responsibilities:
        1. parse the given data request and stored internally into resource
        2. validate if the given data request is valid

        Example data_request
        data_request = {
            "scope": ft.onto.Sentence,
            "schemes": {
                "text_tag": {
                    "entry": ft.onto.Token,
                    "repr": "text_repr",
                    "conversion_method": "indexing",
                    "extractor": AttributeExtractor
                },
                "char_tag": {
                    "entry": ft.onto.Token,
                    "repr": "char_repr",
                    "conversion_method": "indexing",
                    "extractor": AttributeExtractor
                },
                "ner_tag": {
                    "entry": ft.onto.EntityMention,
                    "attribute": "ner_type",
                    "based_on": ft.onto.Token,
                    "strategy": "BIO",
                    "conversion_method": "indexing",
                    "extractor": AnnotationSeqExtractor
                }
            }
        }
        """

        assert "scope" in data_request, \
            "Field not found for data request: `scope`"
        assert "schemes" in data_request, \
            "Field not found for data request: `schemes`"

        self._user_request: Dict = data_request
        self._feature_resource.clear()
        self._feature_resource["scope"] = data_request["scope"]

        resource_schemes = {}
        # Used for check dependency between different extractors
        scheme_group = {
            "dependent": {}, "dependee": {}
        }

        for tag, scheme in data_request["schemes"].items():
            assert "extractor" in scheme, \
                "Field not found for data request scheme: `extractor`"
            assert "type" in scheme, \
                "Field not found for data request scheme: `type`"
            resource_schemes[tag] = {}

            # Build config and extractor
            config = {}
            for field, value in scheme.items():
                if field != "extractor":
                    config[field] = value

            if not issubclass(scheme["extractor"], BaseExtractor):
                raise RuntimeError("Invalid extractor class: "
                                   , scheme["extractor"])

            try:
                extractor: BaseExtractor = scheme["extractor"](config)
                resource_schemes[tag]["extractor"] = extractor

                # Track dependency
                if hasattr(extractor, "based_on"):
                    if extractor.entry_type not in scheme_group["dependent"]:
                        scheme_group["dependent"][extractor.entry_type] = set()
                    scheme_group["dependent"][extractor.entry_type].add(
                        extractor)
                else:
                    if extractor.entry_type not in scheme_group["dependee"]:
                        scheme_group["dependee"][extractor.entry_type] = set()
                    scheme_group["dependee"][extractor.entry_type].add(
                        extractor)

                need_pad: bool = \
                    scheme["need_pad"] if "need_pad" in scheme else True
                converter: Converter = Converter(need_pad=need_pad)
                resource_schemes[tag]["converter"] = converter
                resource_schemes[tag]["type"] = scheme["type"]

                if scheme["type"] == DATA_OUTPUT:
                    unpadder: BaseUnpadder = \
                        unpadder_selector(scheme["strategy"])(config)
                    resource_schemes[tag]["unpadder"] = unpadder

            except Exception as e:
                logger.error("Error instantiate extractor: " + str(e))
                raise

        # Check dependency
        for _, dependent_extractors in scheme_group["dependent"].items():
            for dependent_extractor in dependent_extractors:
                based_on: Entry = dependent_extractor.based_on
                if based_on not in scheme_group["dependee"]:
                    raise "Cannot found based on entry {} for extractor {}". \
                        format(based_on, dependent_extractor.tag)

        self._feature_resource["schemes"] = resource_schemes

    def _build_vocab(self):
        scope: EntryType = self._feature_resource["scope"]
        schemes: Dict = self._feature_resource["schemes"]

        # TODO: clear vocab?
        for data_pack in self._train_data_pack_loader.load():
            for instance in data_pack.get(scope):
                for tag, scheme in schemes.items():
                    extractor: BaseExtractor = scheme["extractor"]
                    extractor.update_vocab(data_pack, instance)

    def _build_dataset_iterator(self, data_pack_loader: DataPackLoader) \
            -> DataIterator:
        scope: Type[EntryType] = self._feature_resource["scope"]
        schemes: Dict[str, Dict[str, Any]] = self._feature_resource["schemes"]

        data_source = \
            DataPackDataSource(data_pack_loader=data_pack_loader,
                               context_type=scope,
                               request={scope: []})

        dataset = DataPackDataset(data_source,
                                  schemes,
                                  self._config.dataset,
                                  self.device)
        iterator = DataIterator(dataset)

        return iterator

    @property
    def feature_resource(self) -> Dict:
        return self._feature_resource

    @property
    def user_request(self) -> Dict:
        return self._user_request

    @property
    def num_epochs(self) -> int:
        return self._config.train.num_epochs

    @property
    def device(self) -> device:
        return self._config.train.device

    @property
    def config(self) -> Config:
        return self._config

    @property
    def state(self) -> Dict:
        return {
            "feature_resource": self._feature_resource,
            "train_config": self._config.train,
            "user_request": self.user_request
        }

    def save_state(self, filename: str):
        torch.save(self.state, filename)

    def save_model(self, filename: str):
        torch.save(self._trainer.model, filename)

    def run(self, data_request):
        # Steps:
        # 1. parse request
        # 2. build vocab
        # 3. for each data pack, do:
        #   Extract + Caching -> Batching -> Padding
        # 4. send batched & padded tensors to trainer

        # Parse and validate data request
        logger.info("Parse user request.")
        self._parse_request(data_request)

        logger.info("Build vocabulary.")
        self._build_vocab()

        # Model can only be initialized after here as it needs the built vocab
        schemes: Dict[str, Dict[str, Any]] = self._feature_resource["schemes"]

        logger.info("Setup trainer.")
        self._trainer.setup(schemes)

        # Setup evaluation pipeline
        if self._config.pipeline.evaluate:
            self._predictor = Predictor(
                batch_size=self._config.dataset.batch_size,
                pretrain_model=self._trainer.model,
                predict_forward_fn=self._predict_forward_fn,
                feature_resource=self._feature_resource,
                cross_pack=False)

        train_iterator = \
            self._build_dataset_iterator(self._train_data_pack_loader)

        logger.info("Start training.")
        epoch = 0
        while epoch < self.num_epochs:
            logger.info("Epoch: %s", epoch)
            epoch += 1

            logger.info("Epoch training")
            train_err = 0.0
            train_total = 0.0
            for batch in tqdm(train_iterator):
                batch_train_err, batch_size = self._trainer.train(batch)
                # post training
                train_err += batch_train_err
                train_total += batch_size

            logger.info("Epoch evaluating")
            # Evaluation will explicitly call loader to iterate over each
            # data pack. The batching will be handled internally by predictor.
            for orig_pack in tqdm(self._val_data_pack_loader.load()):
                predicted_pack = orig_pack.view()
                self._predictor.predict(predicted_pack)
                self._evaluator.consume_next(predicted_pack, orig_pack)

            logger.info("eval result: %s", self._evaluator.get_result())

        self.finish()

    def finish(self):
        self._train_data_pack_loader.finish()
        self._val_data_pack_loader.finish()


def unpadder_selector(encode_strategy: str) -> Type[BaseUnpadder]:
    mapping = {
        "BIO": SameLengthUnpadder
    }

    if encode_strategy not in mapping:
        raise ValueError("Cannot suitable unpadder for encode strategy: "
                         + encode_strategy)

    return mapping[encode_strategy]
