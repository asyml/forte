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

from forte.data.data_pack import DataPack

from forte.data.extractor.predictor import Predictor
from texar.torch.data import DataIterator, Batch
import torch
from torch import device
from typing import Optional, Dict, Type, Any, Union, Iterator

from forte.data.extractor.unpadder import BaseUnpadder, SameLengthUnpadder
from forte.data.types import DATA_OUTPUT
from forte.data.extractor.data_pack_loader import DataPackLoader
from forte.data.extractor.data_pack_dataset import DataPackDataSource, \
    DataPackDataset
from forte.data.extractor.converter import Converter
from forte.common.configuration import Config
from forte.data.ontology.core import EntryType
from forte.data.extractor.extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.data.readers.base_reader import PackReader

logger = logging.getLogger(__name__)


class TrainPreprocessor:
    """
    `TrainPreprocessor` provides the functionality of doing pre-processing work
    including building vocabulary, extracting the features, batching and
    padding. It will provide methods to return the iterator over batch of
    padded tensors.

    `TrainPreprocessor` will maintain a Config that stores all the configurable
    parameters for various components. In addition, those
    parameters are organized in a hierarchical way.

    For example, at first level, it will contain: pipeline, data_pack and
    dataset. The `default_config` will return the default value for all
    configurable parameters.

    Particularly, the `dataset` config is exactly the same as what
    texar.torch.data.DatasetBase::default_config will return.

    `TrainPreprocessor` will also accept a user request. Internally it will
    parse this user request and store the parsed result as a dictionary
    into `feature_resource`.
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
                 request: Dict,
                 config: Optional[Union[Config, Dict]] = None):
        self._config: Config = \
            Config(config, default_hparams=self.default_configs())
        self._validate_config()

        self._train_reader: PackReader = train_reader
        self._val_reader: PackReader = val_reader
        self._predictor: Optional[Predictor] = None

        self._train_data_pack_loader: DataPackLoader = \
            DataPackLoader(reader=self._train_reader,
                           cache_dir=self._config.data_pack.train_cache_dir,
                           config=self._config.data_pack.train_loader)

        self._val_data_pack_loader: DataPackLoader = \
            DataPackLoader(reader=self._val_reader,
                           cache_dir=self._config.data_pack.val_cache_dir,
                           config=self._config.data_pack.val_loader)

        self._user_request: Dict = request
        self._feature_resource: Dict = {}

        if not self._config.pipeline.lazy_parse_request:
            self._parse_request(self._user_request)

        if not self._config.pipeline.lazy_build_vocab:
            assert not self._config.pipeline.lazy_parse_request
            self._build_vocab()

    @staticmethod
    def default_configs():
        # Configs should be serializable
        return {
            "pipeline": {
                "lazy_parse_request": False,
                "lazy_build_vocab": False,
                "device": "cpu",
            },
            "data_pack": {
                "train_loader": DataPackLoader.default_configs(),
                "train_cache_dir": ".train_data_pack_cache",
                "val_loader": DataPackLoader.default_configs(),
                "val_cache_dir": ".val_data_pack_cache"
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
    def device(self) -> device:
        return torch.device(self._config.pipeline.device)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def state(self) -> Dict:
        return {
            "feature_resource": self._feature_resource,
            "user_request": self.user_request,
            "configs": self._config
        }

    def save_state(self, filename: str):
        torch.save(self.state, filename)

    def get_train_pack_iterator(self) -> Iterator[DataPack]:
        return self._train_data_pack_loader.load()

    def get_val_pack_iterator(self) -> Iterator[DataPack]:
        return self._val_data_pack_loader.load()

    def get_train_batch_iterator(self) -> Iterator[Batch]:
        if self.config.pipeline.lazy_parse_request:
            self._parse_request(self._user_request)
        else:
            assert self._feature_resource, "Feature recourse is not parsed"

        if self._config.pipeline.lazy_build_vocab:
            self._build_vocab()
        else:
            assert self._feature_resource, "Feature recourse is not parsed"

        dataset_iter = \
            self._build_dataset_iterator(self._train_data_pack_loader)

        return iter(dataset_iter)

    def get_val_batch_iterator(self) -> Iterator[Batch]:
        if self.config.pipeline.lazy_parse_request:
            self._parse_request(self._user_request)
        else:
            assert self._feature_resource, "Feature recourse is not parsed"

        # Assume vocab is already built

        dataset_iter = \
            self._build_dataset_iterator(self._val_data_pack_loader)

        return iter(dataset_iter)

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
