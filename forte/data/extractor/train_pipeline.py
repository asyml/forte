#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
from typing import Optional, Dict, List, Type, Iterator, Tuple, Any

import numpy as np
import torchtext
import torch
from torch import Tensor, device

from data.extractor.converter import Converter
from forte.data.extractor.feature import Feature
from forte.common.configuration import Config
from forte.data.extractor.trainer import Trainer
from forte.data.ontology.core import EntryType

from forte.data.base_pack import BasePack

from forte.data.extractor.extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.evaluation.base import Evaluator
from forte.data.readers.base_reader import BaseReader

logger = logging.getLogger(__name__)


# TODO: BasePack or DataPack

class TrainPipeline:
    def __init__(self,
                 train_reader: BaseReader,
                 dev_reader: BaseReader,
                 trainer: Trainer,
                 train_path: str,
                 num_epochs: int,
                 batch_size: int,
                 evaluator: Optional[Evaluator] = None,
                 val_path: Optional[str] = None,
                 device_: Optional[device] = torch.device("cpu")):
        """
        Example resource format:
        resource = {
              "scope": ft.onto.Sentence,
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
              },
              "converter": Converter
        }
        """
        self.train_reader = train_reader
        self.dev_reader = dev_reader

        self.trainer: Trainer = trainer
        self.train_path = train_path

        self.evaluator = evaluator
        self.val_path = val_path

        self.resource: Dict[str, Any] = {}

        self.config = Config({}, default_hparams=None)
        self.config.add_hparam('num_epochs', num_epochs)
        self.config.add_hparam('batch_size', batch_size)
        self.config.add_hparam('device', device_)

    def parse_request(self, data_request):
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

        self.resource["converter"] = Converter()
        self.resource["scope"] = data_request["scope"]

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
                    raise "Cannot found based on entry {} for extractor {}".\
                        format(based_on, dependent_extractor.tag)

        self.resource["schemes"] = resource_schemes

    def run(self, data_request):
        # Steps:
        # 1. parse request
        # 2. build vocab
        # 3. for each data pack, do:
        #   Extract -> Caching -> Batching -> Padding
        # 4. send batched & padded tensors to trainer

        # Parse and validate data request
        self.parse_request(data_request)

        # TODO: read all data packs is not memory friendly. Probably should
        #  cache data pack when retrieve it multiple times
        for data_pack in self.train_reader.iter(self.train_path):
            self.build_vocab(data_pack)

        # Model can only be initialized after here as it needs the built vocab
        schemes: Dict[str, Dict[str, BaseExtractor]] = self.resource["schemes"]
        self.trainer.setup(schemes)

        # TODO: evaluation setup
        if self.evaluator:
            pass

        epoch = 0
        while epoch < self.config.num_epochs:
            print("Train epoch:", epoch)

            epoch += 1

            for data_pack in self.train_reader.iter(self.train_path):
                for batch_feature_collection in self.extract(data_pack,
                                                             self.config.batch_size):

                    # TODO: should we:
                    # 1) mask tensor be a member in Feature class and pass
                    # features to user
                    # or
                    # 2) we explicitly put mask tensor and feature tensor into
                    # a dictionary and pass this dict to user.
                    # Note: they are passed to user via method:
                    #       pass_tensor_to_model_fn())
                    #       Currently, we choose the former approach.
                    batch_tensor_collection = \
                        self.convert(batch_feature_collection)
                    self.trainer.train(batch_tensor_collection)

                    # TODO: evaluation process
                    if self.evaluator:
                        pass

    def build_vocab(self, data_pack: BasePack):
        scope: EntryType = self.resource["scope"]
        schemes: Dict = self.resource["schemes"]

        for instance in data_pack.get(scope):
            for tag, scheme in schemes.items():
                extractor: BaseExtractor = scheme["extractor"]
                extractor.update_vocab(data_pack, instance)

    # Extract should extract a single data pack. It should batch and shuffle
    # all the extracted data and return a generator for a batch of extracted
    # tensors as the tensor_collection
    # Example return format
    # """
    # tensor_collection = {
    #         "text_tag": {
    #             "tensor": [<tensor>, <tensor>],
    #             "mask": [<tensor>, <tensor>]
    #         },
    #         "char_tag": {
    #             "tensor": [<tensor>, <tensor>],
    #             "mask": [<tensor>, <tensor>]
    #         },
    #         "ner_tag": {
    #             "tensor": [<tensor>, <tensor>],
    #             "mask": [<tensor>, <tensor>]
    #         }
    # }
    # """
    def extract(self, data_pack: BasePack, batch_size: int) -> \
            Iterator[Dict[str, List[Feature]]]:
        scope: Type[EntryType] = self.resource["scope"]
        schemes: Dict = self.resource["schemes"]
        batch_feature_collection: Dict[str, List[Feature]] = {}

        instances = list(data_pack.get(scope))

        # Extract all instances
        feature_collection: List[Dict[str, Feature]] = []

        for instance in instances:
            feature_collection.append({})
            for tag, scheme in schemes.items():
                extractor: BaseExtractor = scheme["extractor"]
                # TODO: read from cache here
                feature: Feature = extractor.extract(data_pack, instance)
                # TODO: store to cache

                feature_collection[-1][tag] = feature

        # random.shuffle(tensors) # TODO: do we need this?
        # TODO: check batch_size_fn
        # TODO: A better tool for doing batching.
        data_iterator = torchtext.data.iterator.pool(
            feature_collection,
            batch_size,
            key=None,  # TODO: check this
            sort_within_batch=False,  # TODO: check this
            shuffle=False  # TODO: check this
        )

        # Yield a batch of features grouped by tag
        for batch_feature in data_iterator:
            batch_feature_collection.clear()
            for tag, scheme in schemes.items():
                batch_feature_collection[tag] = []

            for feature_by_tag in batch_feature:
                for tag, feature in feature_by_tag.items():
                    batch_feature_collection[tag].append(feature)

            yield batch_feature_collection

    def convert(self, batch_feature_collection: Dict[str, List[Feature]]) -> \
            Dict[str, Dict[str, Tensor]]:
        tensor_collection: Dict[str, Dict[str, Tensor]] = {}

        for tag, features in batch_feature_collection.items():
            converter: Converter = self.resource["schemes"][tag]["converter"]
            tensor, mask = converter.convert(features)
            tensor_collection[tag]["tensor"] = tensor
            tensor_collection[tag]["mask"] = mask

        return tensor_collection
