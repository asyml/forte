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
from typing import Optional, Dict, Type, Any, Union, Iterator

import torch
from texar.torch.data import DataIterator, Batch
from torch import device

from forte.common.configuration import Config
from forte.data.converter.converter import Converter
from forte.data.data_pack_dataset import DataPackDataSource, \
    DataPackDataset
from forte.data.extractor.base_extractor import BaseExtractor
from forte.predictor import Predictor
from forte.data.converter.unpadder import BaseUnpadder, DefaultUnpadder
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType
from forte.data.readers.base_reader import PackReader
from forte.data.types import DATA_OUTPUT

logger = logging.getLogger(__name__)


class TrainPreprocessor:
    """
    `TrainPreprocessor` provides the functionality of doing pre-processing work
    including building vocabulary, extracting the features, batching and
    padding. It will provide methods to return the iterator over batch of
    padded tensors.

    `TrainPreprocessor` will maintain a Config that stores all the configurable
    parameters for various components.

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
                "type": DATA_INPUT,
                "need_pad": True
            },
            "char_tag" {
                "extractor":  Extractor,
                "converter": Converter,
                "type": DATA_INPUT,
                "need_pad": True
            }
            "ner_tag": {
                "extractor":  Extractor,
                "converter": Converter,
                "unpadder": Unpadder,
                "type": DATA_OUTPUT,
                "need_pad": True
            }
        }
    }
    """

    def __init__(self,
                 train_reader: PackReader,
                 request: Dict,
                 config: Optional[Union[Config, Dict]] = None):
        self._config: Config = \
            Config(config, default_hparams=self.default_configs())
        self._validate_config()

        self._train_reader: PackReader = train_reader
        self._predictor: Optional[Predictor] = None

        self._user_request: Dict = request
        self._feature_resource: Dict = {}
        self._feature_resource_ready: bool = False
        self._vocab_ready: bool = False

        if not self._config.preprocess.lazy_parse_request:
            self._parse_request(self._user_request)

        if not self._config.preprocess.lazy_build_vocab:
            assert not self._config.preprocess.lazy_parse_request
            self._build_vocab()

    @staticmethod
    def default_configs():
        # Configs should be serializable
        return {
            "preprocess": {
                "pack_dir": "",
                "lazy_parse_request": False,
                "lazy_build_vocab": False,
                "device": "cpu",
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
                    "vocab_method": "indexing",
                    "extractor": AttributeExtractor,
                    "type": DATA_INPUT,
                    "need_pad": True
                },
                "char_tag": {
                    "entry": ft.onto.Token,
                    "repr": "char_repr",
                    "vocab_method": "indexing",
                    "extractor": AttributeExtractor,
                    "type": DATA_INPUT,
                    "need_pad": True
                },
                "ner_tag": {
                    "entry": ft.onto.EntityMention,
                    "attribute": "ner_type",
                    "based_on": ft.onto.Token,
                    "strategy": "BIO",
                    "vocab_method": "indexing",
                    "extractor": AnnotationSeqExtractor,
                    "type": DATA_OUTPUT,
                    "need_pad": True
                }
            }
        }
        """

        assert "scope" in data_request, \
            "Field not found for data request: `scope`"
        assert "schemes" in data_request, \
            "Field not found for data request: `schemes`"

        self._user_request = data_request
        self._feature_resource.clear()
        self._feature_resource["scope"] = data_request["scope"]

        resource_schemes: Dict[str, Dict] = {}
        # Used for check dependency between different extractors
        scheme_group: Dict[str, Dict] = {
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
                raise RuntimeError("Invalid extractor class: ",
                                   scheme["extractor"])

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
                converter: Converter = Converter()
                resource_schemes[tag]["converter"] = converter
                resource_schemes[tag]["type"] = scheme["type"]
                resource_schemes[tag]["need_pad"] = need_pad

                if scheme["type"] == DATA_OUTPUT:
                    if "unpadder" in scheme:
                        unpadder: BaseUnpadder = \
                            scheme["unpadder"](config)
                    else:
                        unpadder: BaseUnpadder = DefaultUnpadder(config)
                    resource_schemes[tag]["unpadder"] = unpadder

            except Exception as e:
                logger.error("Error instantiate extractor: %s", str(e))
                raise

        # Check dependency
        for _, dependent_extractors in scheme_group["dependent"].items():
            for dependent_extractor in dependent_extractors:
                based_on: Entry = dependent_extractor.based_on
                if based_on not in scheme_group["dependee"]:
                    raise ValueError(
                        "Cannot found based on entry {} for extractor {}".
                        format(based_on, dependent_extractor.tag))

        self._feature_resource["schemes"] = resource_schemes
        self._feature_resource_ready = True

    def _build_vocab(self):
        scope: EntryType = self._feature_resource["scope"]
        schemes: Dict = self._feature_resource["schemes"]

        # TODO: clear vocab?
        for data_pack in \
                self._train_reader.iter(self._config.preprocess.pack_dir):
            for instance in data_pack.get(scope):
                for _, scheme in schemes.items():
                    extractor: BaseExtractor = scheme["extractor"]
                    if extractor.config.vocab_method != "raw":
                        extractor.update_vocab(data_pack, instance)

        self._vocab_ready = True

    def _build_dataset_iterator(self) \
            -> DataIterator:
        scope: Type[EntryType] = self._feature_resource["scope"]  # type: ignore
        schemes: Dict[str, Dict[str, Any]] = self._feature_resource["schemes"]

        data_source = \
            DataPackDataSource(reader=self._train_reader,
                               pack_dir=self._config.preprocess.pack_dir,
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
        if not self._feature_resource_ready:
            self._parse_request(self._user_request)
        return self._feature_resource

    @property
    def user_request(self) -> Dict:
        return self._user_request

    @property
    def device(self) -> device:
        return torch.device(self._config.preprocess.device)

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

    def get_train_batch_iterator(self) -> Iterator[Batch]:
        if self._config.preprocess.lazy_parse_request:
            self._parse_request(self._user_request)
        else:
            assert self._feature_resource_ready, \
                "Feature recourse is not parsed"

        if self._config.preprocess.lazy_build_vocab:
            self._build_vocab()
        else:
            assert self._vocab_ready, "Vocab is not built"

        dataset_iter = self._build_dataset_iterator()

        return iter(dataset_iter)


def unpadder_selector(encode_strategy: str) -> Type[BaseUnpadder]:
    mapping = {
        "BIO": SameLengthUnpadder
    }

    if encode_strategy not in mapping:
        raise ValueError("Cannot suitable unpadder for encode strategy: "
                         + encode_strategy)

    return mapping[encode_strategy]
