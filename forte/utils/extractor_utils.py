#  Copyright 2021 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pickle
from typing import Dict, Any

from forte.common.configuration import Config
from forte.data import BaseExtractor
from forte.data.converter import Converter
from forte.utils import get_class

DATA_INPUT = 0
DATA_OUTPUT = 1


def parse_feature_extractors(scheme_configs: Config) -> Dict[str, Any]:
    feature_requests: Dict[str, Any] = {}

    for tag, scheme_config in scheme_configs.items():
        assert (
            "extractor" in scheme_config
        ), "Field not found for data request scheme: `extractor`"
        assert (
            "type" in scheme_config
        ), "Field not found for data request scheme: `type`"
        assert scheme_config["type"] in [
            "data_input",
            "data_output",
        ], "Type field must be either data_input or data_output."

        feature_requests[tag] = {}

        if scheme_config["type"] == "data_input":
            feature_requests[tag]["type"] = DATA_INPUT
        elif scheme_config["type"] == "data_output":
            feature_requests[tag]["type"] = DATA_OUTPUT

        extractor_class = get_class(scheme_config["extractor"]["class_name"])
        extractor: BaseExtractor = extractor_class()
        if not isinstance(extractor, BaseExtractor):
            raise RuntimeError(
                "Invalid extractor: ", scheme_config["extractor"]
            )

        extractor.initialize(config=scheme_config["extractor"]["config"])

        # Load vocab from disk if provided.
        if "vocab_path" in scheme_config["extractor"]:
            with open(
                scheme_config["extractor"]["vocab_path"], "rb"
            ) as vocab_file:
                extractor.vocab = pickle.load(vocab_file)

        feature_requests[tag]["extractor"] = extractor

        if "converter" not in scheme_config:
            # Create default converter if there is no given converter
            feature_requests[tag]["converter"] = Converter({})
        else:
            converter_class = get_class(
                scheme_config["converter"]["class_name"]
            )
            converter: Converter = converter_class()
            if not isinstance(converter, Converter):
                raise RuntimeError(
                    "Invalid converter: ", scheme_config["converter"]
                )
            feature_requests[tag]["converter"] = converter

    return feature_requests
