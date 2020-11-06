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
import pickle
import time
from typing import Optional, Dict, List, Type, Any

import torch
from texar.torch.data import DataIterator

from forte.data.extractor.data_pack_dataset import DataPackDataSource, \
    DataPackDataset
from forte.processors.base.base_processor import BaseProcessor

from forte.data.data_utils_io import slice_batch

from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from torch import Tensor, device

from forte.data.extractor.converter import Converter
from forte.data.extractor.feature import Feature
from forte.common.configuration import Config
from forte.data.extractor.trainer import Trainer
from forte.data.ontology.core import EntryType

from forte.data.base_pack import BasePack

from forte.data.extractor.extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.evaluation.base import Evaluator
from forte.data.readers.base_reader import BaseReader, PackReader
from texar.torch import HParams

logger = logging.getLogger(__name__)


# TODO: BasePack or DataPack

class TrainPipeline:
    """
    # TODO:
    """

    def __init__(self,
                 train_reader: BaseReader,
                 dev_reader: BaseReader,
                 trainer: Trainer,
                 train_path: str,
                 num_epochs: int,
                 batch_size: int,
                 predictor: Optional[BaseProcessor] = None,
                 evaluator: Optional[Evaluator] = None,
                 val_path: Optional[str] = None,
                 device_: Optional[device] = torch.device("cpu")):
        """
        `TrainPipeline` will maintain a Config that stores all the configurable
        parameters for various components inside TrainPipeline. In addition,
        those parameters are organized in a hierarchical way. For example, at
        first level, it will contain three parts: pipeline, train and model
        corresponding to pipeline-related, training-related and
        model-hyper-parameter-related parameters. The following example
        demonstrate a typical content of a Config.
        Example config:
        pipeline:
            train_reader:   Reader
            dev_reader:     Reader
            trainer:        Trainer
            predictor:      BatchProcessor
            evaluator:      Evaluator
            feature_resource: {
                "scope": ft.onto.Sentence
                "schemes": {
                    "text_tag": {
                        "extractor":  Extractor,
                        "converter": Converter
                    },
                    "char_tag" {
                        "extractor":  Extractor,
                        "converter": Converter
                    }
                    "ner_tag": {
                        "extractor":  Extractor,
                        "converter": Converter
                    }
                }
            }
        train:
            batch_size: int
            num_epochs: int
            train_path: str
            dev_path:   str
            device:     [device("cpu") | device("cuda")]
        Specifically, there will be a Config.pipeline.feature_resource which 
        indicates feature-related processing parameters such as 
        Extractor, Converter. Those Extractor and Converter are grouped by 
        corresponding tags.
        """
        self.train_reader = train_reader
        self.dev_reader = dev_reader

        self.trainer: Trainer = trainer
        self.train_path = train_path

        self.predictor = predictor

        self.evaluator = evaluator
        self.val_path = val_path

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device_ = device_

        self.batcher: ProcessingBatcher = FixedSizeDataPackBatcher()

        self.feature_resource: Dict[str, Any] = {}

        self._build_config()

    def _build_config(self):
        self.config = Config({}, None)
        pipeline_config = Config({}, None)
        train_config = Config({}, None)
        self.config.add_hparam("pipeline", pipeline_config)
        self.config.add_hparam("train", train_config)

        # Pipeline config
        pipeline_config.add_hparam("train_reader", self.train_reader)
        pipeline_config.add_hparam("dev_reader", self.dev_reader)
        pipeline_config.add_hparam("trainer", self.trainer)
        pipeline_config.add_hparam("predictor", self.predictor)
        pipeline_config.add_hparam("evaluator", self.evaluator)
        pipeline_config.add_hparam("feature_resource",
                                   self.feature_resource)

        # Train config
        train_config.add_hparam("num_epochs", self.num_epochs)
        train_config.add_hparam("batch_size", self.batch_size)
        train_config.add_hparam("train_path", self.train_path)
        train_config.add_hparam("val_path", self.val_path)
        train_config.add_hparam("device", self.device_)

    def _parse_request(self, data_request):
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

        self.feature_resource.clear()
        self.feature_resource["scope"] = data_request["scope"]

        resource_schemes = {}
        # Used for check dependency between different extractors
        scheme_group = {
            "dependent": {}, "dependee": {}
        }

        for tag, scheme in data_request["schemes"].items():
            assert "extractor" in scheme, \
                "Field not found for data request scheme: `extractor`"

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
                    if extractor.entry not in scheme_group["dependent"]:
                        scheme_group["dependent"][extractor.entry] = set()
                    scheme_group["dependent"][extractor.entry].add(extractor)
                else:
                    if extractor.entry not in scheme_group["dependee"]:
                        scheme_group["dependee"][extractor.entry] = set()
                    scheme_group["dependee"][extractor.entry].add(extractor)

                need_pad: bool = \
                    scheme["need_pad"] if "need_pad" in scheme else True
                converter: Converter = Converter(need_pad=need_pad)
                resource_schemes[tag]['converter'] = converter
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

        self.feature_resource["schemes"] = resource_schemes

    def _build_vocab(self):
        scope: EntryType = self.feature_resource["scope"]
        schemes: Dict = self.feature_resource["schemes"]

        # TODO: read all data packs is not memory friendly. Probably should
        #  cache data pack when retrieve it multiple times
        # TODO: clear vocab?
        for data_pack in self.train_reader.iter(self.train_path):
            for instance in data_pack.get(scope):
                for tag, scheme in schemes.items():
                    extractor: BaseExtractor = scheme["extractor"]
                    extractor.update_vocab(data_pack, instance)

    def _build_dataset_iterator(self, file_path: str, reader: BaseReader) \
            -> DataIterator:
        scope: Type[EntryType] = self.feature_resource["scope"]
        schemes: Dict[str, Dict[str, Any]] = self.feature_resource["schemes"]

        data_source = DataPackDataSource(file_path,
                                         reader,
                                         scope,
                                         {scope: []})

        hparams = {"batch_size": self.batch_size}
        dataset = DataPackDataset(data_source,
                                  schemes,
                                  hparams,
                                  self.device_)
        iterator = DataIterator(dataset)

        return iterator

    @staticmethod
    def _get_default_filename() -> str:
        # TODO: file path
        timestamp = str(int(time.time()))
        return timestamp + ".pkl"

    def get_state(self) -> Dict:
        return {
            "feature_resource": self.feature_resource,
            "train_config": self.config.train
        }

    def save_state(self, filename: str):
        if not filename:
            filename = self._get_default_filename()

        with open(filename, 'wb') as f:
            pickle.dump(self.get_state(), f)

    def run(self, data_request):
        # Steps:
        # 1. parse request
        # 2. build vocab
        # 3. for each data pack, do:
        #   Extract + Caching -> Batching -> Padding
        # 4. send batched & padded tensors to trainer

        # Parse and validate data request
        self._parse_request(data_request)

        self._build_vocab()

        # Model can only be initialized after here as it needs the built vocab
        schemes: Dict[str, Dict[str, Any]] = self.feature_resource["schemes"]
        self.trainer.setup(schemes)

        # TODO: evaluation setup
        if self.evaluator:
            pass

        train_iterator = self._build_dataset_iterator(self.train_path,
                                                      self.train_reader)

        epoch = 0
        while epoch < self.num_epochs:
            epoch += 1

            for batch in train_iterator:
                self.trainer.train(batch)

            # TODO: evaluation process
        #     if self.evaluator:
        #         pass
