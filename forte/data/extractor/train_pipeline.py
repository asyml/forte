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
from typing import Optional, Dict, List, Type, Iterator, Tuple

import numpy as np
import torchtext
import torch
from torch import Tensor, device

from common.configuration import Config
from data.extractor.trainer import Trainer
from forte.data.ontology.core import EntryType

from forte.data.base_pack import BasePack

from data.extractor.extractor import BaseExtractor
from data.ontology.core import Entry
from evaluation.base import Evaluator
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
                 device: Optional[device] = torch.device("cpu")):
        self.train_reader = train_reader
        self.dev_reader = dev_reader

        self.trainer: Trainer = trainer
        self.train_path = train_path

        self.evaluator = evaluator
        self.val_path = val_path

        self.resource = {}
        self.config = Config({}, default_hparams=None)
        self.config.add_hparam('num_epochs', num_epochs)
        self.config.add_hparam('batch_size', batch_size)
        self.config.add_hparam('device', device)

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

        Example resource format:
        resource = {
              "scope": ft.onto.Sentence,
              "schemes": {
                  "text_tag": {
                      "extractor":  Extractor
                  },
                  "char_tag" {
                      "extractor":  Extractor
                  }
                  "ner_tag": {
                      "extractor":  Extractor
                  }
              }
        }
        """

        assert "scope" in data_request, \
            "Field not found for data request: `scope`"
        assert "schemes" in data_request, \
            "Field not found for data request: `schemes`"

        self.resource["scope"] = data_request["scope"]

        schemes = {}
        # Used for check dependency between different extractors
        scheme_group = {
            "dependent": {}, "dependee": {}
        }

        for tag, scheme in data_request["schemes"].items():
            assert "extractor" in scheme, \
                "Field not found for data request scheme: `extractor`"

            schemes[tag] = {}

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
                schemes[tag]["extractor"] = extractor

                # Track dependency
                if hasattr(extractor, "based_on"):
                    if extractor.entry not in scheme_group["dependent"]:
                        scheme_group["dependent"][extractor.entry] = set()
                    scheme_group["dependent"][extractor.entry].add(extractor)
                else:
                    if extractor.entry not in scheme_group["dependee"]:
                        scheme_group["dependee"][extractor.entry] = set()
                    scheme_group["dependee"][extractor.entry].add(extractor)
            except Exception as e:
                logger.error("Error instantiate extractor: " + str(e))
                raise

        # Check dependency
        for _, dependent_extractors in scheme_group["dependent"].items():
            for dependent_extractor in dependent_extractors:
                based_on: Entry = dependent_extractor.based_on
                if based_on not in scheme_group["dependee"]:
                    raise "Cannot found based on entry {} for extractor {}".format(
                        based_on, dependent_extractor.tag
                    )

        self.resource["schemes"] = schemes

    def run(self, data_request):
        # Parse and validate data request
        self.parse_request(data_request)

        extractor_handler = ExtractorHandler(self.resource, self.config)

        data_packs = list(
            self.train_reader.iter(self.train_path))

        extractor_handler.build_vocab(data_packs)

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

            for data_pack in data_packs:
                for batch in extractor_handler.extract(data_pack,
                                                       self.config.batch_size):
                    self.trainer.train(batch)

                    # TODO: evaluation process
                    if self.evaluator:
                        pass


class ExtractorHandler():
    def __init__(self, resource: Dict, config: Config):
        self.resource = resource
        self.config = config

    def build_vocab(self, data_packs: List[BasePack]):
        scope: EntryType = self.resource["scope"]
        schemes: Dict = self.resource["schemes"]

        for data_pack in data_packs:
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
            Iterator[Dict[str, Dict[str, Tensor]]]:
        scope: Type[EntryType] = self.resource["scope"]
        schemes: Dict = self.resource["schemes"]
        tensor_collection: Dict[str, Dict[str, Tensor]] = {}

        instances = list(data_pack.get(scope))

        # Extract all instances
        tensor_list = []

        for instance in instances:
            tensor_list.append({})
            for tag, scheme in schemes.items():
                extractor: BaseExtractor = scheme["extractor"]
                tensor = extractor.extract(data_pack, instance)

                tensor_list[-1][tag] = tensor

        # random.shuffle(tensors) # TODO: do we need this?
        # TODO: check batch_size_fn
        data_iterator = torchtext.data.iterator.pool(
            tensor_list,
            batch_size,
            key=None,  # TODO: check this
            sort_within_batch=False,  # TODO: check this
            shuffle=False  # TODO: check this
        )

        for tag, scheme in schemes.items():
            tensor_collection[tag] = {}

        # Yield each extracted and padded batch
        for batch in data_iterator:
            unpadded_tensor_collection = {}
            for extracted_instance in batch:
                for tag, tensor in extracted_instance.items():
                    if tag not in unpadded_tensor_collection:
                        unpadded_tensor_collection[tag] = []
                    unpadded_tensor_collection[tag].append(tensor)

            # TODO: padding should probably be a dedicated class
            # TODO: there should be no hardcoding for tag
            for tag, unpadded_tensors in unpadded_tensor_collection.items():
                extractor: BaseExtractor = schemes[tag]["extractor"]
                padded_tensors, masks = \
                    self.padding(unpadded_tensors, extractor.get_pad_id(), tag)
                tensor_collection[tag]["tensor"] = padded_tensors
                tensor_collection[tag]["mask"] = masks

            yield tensor_collection

    def padding(self, tensors: List[Tensor], pad_id: int, tag: str) \
            -> Tuple[Tensor, Tensor]:
        if tag != "char_tag":
            batch_size = len(tensors)
            max_length = max([len(t) for t in tensors])

            padded_tensors = np.empty([batch_size, max_length], dtype=np.int64)
            masks = np.zeros([batch_size, max_length], dtype=np.float32)

            for i, tensor in enumerate(tensors):
                curr_len = len(tensor)

                padded_tensors[i, :curr_len] = tensor
                padded_tensors[i, curr_len:] = pad_id

                masks[i, :curr_len] = 1.0

            padded_tensors = torch.from_numpy(padded_tensors).to(self.config.device)
            masks = torch.from_numpy(masks).to(self.config.device)

            return padded_tensors, masks
        else:
            batch_size = len(tensors)
            max_length = max([len(t) for t in tensors])
            char_length = max(
                [max([len(charseq) for charseq in d]) for d in tensors]
            )

            padded_tensors = np.empty([batch_size, max_length, char_length],
                                      dtype=np.int64)
            masks = np.zeros([batch_size, max_length], dtype=np.float32)

            for i, tensor in enumerate(tensors):
                curr_len = len(tensor)

                for c, cids in enumerate(tensor):
                    padded_tensors[i, c, :len(cids)] = cids
                    padded_tensors[i, c, len(cids):] = pad_id
                padded_tensors[i, curr_len:, :] = pad_id

                masks[i, :curr_len] = 1.0

            padded_tensors = torch.from_numpy(padded_tensors).to(
                self.config.device)
            masks = torch.from_numpy(masks).to(self.config.device)

            return padded_tensors, masks
