# Copyright 2019 The Forte Authors. All Rights Reserved.
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
"""
The processors that process data in batch.
"""
from typing import Dict, Optional, Any

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.base_pack import PackType
from forte.data.batchers import (
    ProcessingBatcher,
)
from forte.data.batchers_w_extractor import (
    FixedSizeDataPackBatcherWithExtractor,
)
from forte.data.converter import Converter
from forte.data.ontology.top import Annotation
from forte.utils import extractor_utils
from forte.processors.base.batch_processor import PackingBatchProcessor
from forte.data.base_extractor import BaseExtractor

__all__ = [
    "Predictor",
]

from forte.utils.extractor_utils import (
    parse_feature_extractors,
)


class Predictor(PackingBatchProcessor[PackType]):
    r"""
    `Predictor` is a special type of batch processor that uses
    :class:`~forte.data.BaseExtractor` to collect features from data packs, and
    also uses Extractors to write the prediction back.

    `Predictor` implements the `PackingBatchProcessor` class, and implements
    the `predict` and `pack` function using the extractors.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.do_eval = False
        self._request: Dict = {}
        self._request_ready: bool = False

        self._batcher: Optional[FixedSizeDataPackBatcherWithExtractor] = None

    def add_extractor(
        self,
        name: str,
        extractor: BaseExtractor,
        is_input: bool,
        converter: Optional[Converter] = None,
    ):
        """
        Extractors can be added to the preprocessor directly via this
        method.

        Args:
            name: The name/identifier of this extractor, the name should be
              different between different extractors.
            extractor: The extractor instance to be added.
            is_input: Whether this extractor will be used as input or output.
            converter:  The converter instance to be applied after running
              the extractor.

        Returns:
            None
        """
        extractor_utils.add_extractor(
            self._request, name, extractor, is_input, converter
        )

    def set_feature_requests(self, request: Dict):
        self._request = request
        self._request_ready = True

    def deactivate_request(self):
        self._request_ready = False

    @classmethod
    def define_batcher(cls) -> ProcessingBatcher:
        return FixedSizeDataPackBatcherWithExtractor()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "feature_scheme": None,
            "context_type": None,
            "batcher": cls.define_batcher().default_configs(),
            "do_eval": False,
        }

    def initialize(self, resources: Resources, configs: Config):
        # Populate the _request. The self._request_ready help avoid parsing
        # the feature scheme multiple times during `initialize`.
        if not self._request_ready:
            for key, value in configs.items():
                if key == "feature_scheme":
                    self._request["schemes"] = parse_feature_extractors(
                        configs.feature_scheme
                    )
                else:
                    self._request[key] = value
            self._request_ready = True

        batcher_config = configs.batcher
        # Assign context type from here to make sure batcher is using the
        # same context type as predictor.
        batcher_context = configs["batcher"].get("context_type", None)
        if (
            batcher_context is None
            or batcher_context == self._request["context_type"]
        ):
            batcher_config.context_type = self._request["context_type"]
        else:
            raise ProcessorConfigError(
                "The 'context_type' configuration value should be the same "
                "for the processor and the batcher, now for the processor the "
                f"value is {self._request['context_type']} and for the "
                f"batcher the value is {batcher_context}. It is also fine if "
                f"this value for batch config is left empty."
            )
        self.do_eval = configs.do_eval

        # This needs to be called later since batcher config needs to be loaded.
        super().initialize(resources, configs)
        for tag, scheme in self._request["schemes"].items():
            # Add input feature to the batcher.
            if scheme["type"] == extractor_utils.DATA_INPUT:
                self.batcher.add_feature_scheme(tag, scheme)  # type: ignore

    def load(self, model):
        self.model = model

    def pack(
        self,
        pack: PackType,
        predict_results: Dict,
        context: Optional[Annotation] = None,
    ):
        for tag, batched_predictions in predict_results.items():
            # preds contains batched results.
            if self.do_eval:
                self.__extractor(tag).pre_evaluation_action(pack, context)
            for prediction in batched_predictions:
                self.__extractor(tag).add_to_pack(pack, prediction, context)
        pack.add_all_remaining_entries()

    def __extractor(self, tag_name: str):
        return self._request["schemes"][tag_name]["extractor"]

    def predict(self, data_batch: Dict) -> Dict:
        r"""The function that task processors should implement. Make
        predictions for the input ``data_batch``.

        Args:
              data_batch: A batch of instances in our ``dict`` format.

        Returns:
              The prediction results in dict datasets.
        """
        raise NotImplementedError()
