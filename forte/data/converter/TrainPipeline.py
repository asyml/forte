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
from typing import Optional

from common.configuration import Config
from data.converter.ConvertHandler import ConvertHandler
from data.converter.converter import BaseConverter
from data.ontology.core import Entry
from evaluation.base import Evaluator
from forte.data.readers.base_reader import BaseReader
from forte.trainer.base.base_trainer import BaseTrainer
from processors.base import BaseProcessor


logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self, train_reader: BaseReader, trainer: BaseTrainer,
                 dev_reader: BaseReader, configs: Config,
                 evaluator: Optional[Evaluator] = None,
                 predictor: Optional[BaseProcessor] = None):
        self.train_reader = train_reader
        self.dev_reader = dev_reader

        self.trainer = trainer

        self.configs = configs
        self.resource = {}

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
                    "converter": OneToOneConverter
                },
                "char_tag" {
                    "entry": ft.onto.Token,
                    "repr": "char_repr",
                    "conversion_method": "indexing",
                    "converter": OneToManyConverter
                },
                "ner_tag": {
                    "entry": ft.onto.EntityMention,
                    "label": "ner_type",
                    "based_on": ft.onto.Token,
                    "strategy": "BIO",
                    "conversion_method": "indexing",
                    "converter": ManyToOneConverter
                }
            }
        }

        Example resource format:
        resource = {
              "scope": ft.onto.Sentence,
              "schemes": {
                  "text_tag": {
                      "converter":  Converter
                  },
                  "char_tag" {
                      "converter":  Converter
                  }
                  "ner_tag": {
                      "converter":  Converter
                  }
              },
              "vocab_collection": { # to be filled during conversion
                  "text_tag": Vocab,
                  "char_tag": Vocab,
                  "ner_tag": Vocab
              }
        }
        """

        assert "scope" in data_request, \
            "Field not found for data request: `scope`"
        assert "schemes" in data_request, \
            "Field not found for data request: `schemes`"

        self.resource["scope"] = data_request["scope"]

        schemes = {}
        # Used for check dependency between different converters
        scheme_group = {
            "dependent": {}, "dependee": {}
        }

        for tag, scheme in data_request["schemes"]:
            assert "converter" in data_request, \
                "Field not found for data request scheme: `converter`"

            schemes[tag] = {}

            # Build config and converter
            config = {}
            for field, value in scheme.items():
                if (field != "converter"):
                    config[field] = value

            if type(scheme["converter"]) != BaseConverter:
                raise RuntimeError("Invalid converter class: "
                                   + scheme["converter"])

            try:
                converter: BaseConverter = scheme["converter"](config)
                schemes[tag]["converter"] = converter

                # Track dependency
                if (hasattr(converter, "based_on")):
                    scheme_group["dependent"][converter.entry] = converter
                else:
                    scheme_group["dependee"][converter.entry] = converter
            except Exception as e:
                logger.error("Error instantiate converter: " + str(e))
                raise

        # Check dependency
        for _, dependent_converter in scheme_group["dependent"].items():
            based_on: Entry = dependent_converter.based_on
            if based_on not in scheme_group["dependee"]:
                raise "Cannot found based on entry {} for converter {}".format(
                    based_on, dependent_converter.tag
                )

        self.resource["schemes"] = schemes

    def run(self, data_request):
        # Parse and validate data request
        self.parse_request(data_request)

        convert_handler = ConvertHandler(self.resource)

        epoch = 0
        while True:
            epoch += 1
            data_packs = list(
                self.train_reader.iter(self.configs.config_data.train_path))

            # Let conversion_handler do the remaining conversion
            tensor_collection = convert_handler.convert(data_packs)

            # Call trainer to pass tensors to model and do the training
            self.trainer.consume(tensor_collection)
            self.trainer.epoch_finish_action(epoch)

            # TODO: evaluation step

            if self.trainer.stop_train():
                return
