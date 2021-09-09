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
"""
Train preprocessor helps doing data pre-processing during training.
"""
import logging
from typing import Optional, Dict, Type, Any, Union, Iterator, List

import torch
from texar.torch.data import DataIterator, Batch
from torch import device

from forte.common.configuration import Config
from forte.data.base_extractor import BaseExtractor
from forte.data.converter import Converter
from forte.data.data_pack import DataPack
from forte.data.data_pack_dataset import DataPackDataset, DataPackIterator
from forte.data.ontology import Annotation
from forte.data.ontology.core import EntryType
from forte.utils import extractor_utils
from forte.utils.extractor_utils import parse_feature_extractors

logger = logging.getLogger(__name__)

__all__ = ["TrainPreprocessor"]


class TrainPreprocessor:
    r"""
    `TrainPreprocessor` provides the functionality of doing pre-processing work
    including building vocabulary, extracting the features, batching and
    padding (optional). The processed data will be provided by its method
    :meth:`get_train_batch_iterator`, which will return an `iterator` over the
    batch of pre-processed data. Please refer to the documentation of
    that method for how the pre-processing is done.

    A main part of the `TrainPreprocessor ` is that it maintains a list of
    extractors :class:`~forte.data.BaseExtractor` that extract features. This
    can be provided either via calling `add_extractor` function. Alternatively,
    a request can be passed in through `initialize`, where the configuration
    under the `request` key will be used to create the extractor instances.

    The parsed components will be stored, and can be accessed via the `request`
    property of this class.

    Args:
        pack_iterator (Iterator[DataPack]): An iterator of
            :class:`~forte.data.data_pack.DataPack`.

    .. note::
        For parameters `request`, user does not necessarily need to provide
        `converter`. If no `converter` is specified, a default converter of
        type :class:`~forte.data.converter.Converter` will be picked.
    """

    DATA_INPUT = extractor_utils.DATA_INPUT
    DATA_OUTPUT = extractor_utils.DATA_OUTPUT

    def __init__(self, pack_iterator: Iterator[DataPack]):
        self._pack_iterator: Iterator[DataPack] = pack_iterator
        self._cached_packs: List[DataPack] = []

        self._config: Config = None
        self._user_request: Dict = {}
        # Parsed feature extractors.
        self._request: Dict = {}
        self._request_ready: bool = False
        self._vocab_ready: bool = False

    def initialize(self, config: Optional[Union[Config, Dict]] = None):
        self._config = Config(
            config,
            default_hparams=self.default_configs(),
            allow_new_hparam=True,
        )
        self._user_request = self._config.request
        self._validate_config()
        self._parse_request(self._user_request)
        self._build_vocab()

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

        """
        extractor_utils.add_extractor(
            self._request, name, extractor, is_input, converter
        )

    @staticmethod
    def default_configs():
        r"""Returns a dictionary of default hyper-parameters.

        .. code-block:: python

            {
                "preprocess": {
                            "device": "cpu",
                },
                "dataset": DataPackDataset.default_hparams()
            }

        Here:

        `"preprocessor.device"`:
            The device of the produced batches. For GPU training,
            set to current CUDA device.

        `"dataset"`:
            This contains all the configurable options same as
            :class:`~forte.data.data_pack_dataset.DataPackDataset`.
        """

        # Configs should be serializable
        return {
            "preprocess": {
                "device": "cpu",
            },
            "dataset": DataPackDataset.default_hparams(),
            "request": {"context_type": None, "feature_scheme": None},
        }

    def _validate_config(self):
        # Placeholder
        pass

    def _parse_request(self, request: Dict):
        """
        This method has two responsibilities:
        1. parse the given data request and stored it internally
        2. validate if the given data request is valid
        """
        parsed_request: Dict[str, Any] = {}

        if "context_type" not in request or request["context_type"] is None:
            raise ValueError("Field not found for data request: `context_type`")

        if "feature_scheme" not in request or request["feature_scheme"] is None:
            raise ValueError(
                "Field not found for data request: `feature_scheme`"
            )

        parsed_request["context_type"] = request["context_type"]
        parsed_request["schemes"] = parse_feature_extractors(
            request["feature_scheme"]
        )

        self._request = parsed_request
        self._request_ready = True

    def _build_vocab(self):
        context_type: EntryType = self._request["context_type"]
        schemes: Dict = self._request["schemes"]

        # TODO: clear vocab?

        # Cached all data packs
        # TODO: this caching is not scalable
        for data_pack in self._pack_iterator:
            self._cached_packs.append(data_pack)

        for _, scheme in schemes.items():
            extractor: BaseExtractor = scheme["extractor"]
            if extractor.vocab_method != "raw":
                for data_pack in self._cached_packs:
                    if context_type is None:
                        extractor.update_vocab(data_pack)
                    else:
                        context: Annotation
                        for context in data_pack.get(context_type):
                            extractor.update_vocab(data_pack, context)

        self._vocab_ready = True

    def _build_dataset_iterator(self) -> DataIterator:
        context_type: Type[EntryType] = self._request["context_type"]  # type: ignore
        schemes: Dict[str, Dict[str, Any]] = self._request["schemes"]

        data_source = DataPackIterator(
            pack_iterator=iter(self._cached_packs),
            context_type=context_type,
            request={context_type: []},
        )

        dataset = DataPackDataset(
            data_source, schemes, self._config.dataset, self.device
        )
        iterator = DataIterator(dataset)

        return iterator

    @property
    def request(self) -> Dict:
        # pylint: disable=line-too-long
        r"""A `Dict` containing all the information needed for doing the
            pre-processing. This is obtained via parsing the input `request`

            An example `request` is:

            .. code-block:: python

                request = {
                    "context_type": "ft.onto.base_ontology.Sentence"
                    "schemes": {
                        "text_tag": {
                            "extractor":
                                "class_name":
                                  "forte.data.extractor.AttributeExtractor",
                                "config": {
                                    ... more configuration of the extractor
                                }
                        },
                        "ner_tag": {
                            "extractor":
                                "class_name":
                                  "forte.data.extractor.BioSeqTaggingExtractor",
                                "config": {
                                    ... more configuration of the extractor
                                }
                        }
                    }
                }

        Here:

            `"context_type"`: Annotation
                A class of type :class:`~ft.onto.base_ontology.context_type`.
                Defines the granularity to separate data into different
                groups. All extractors will operate based on this. For example,
                if `context_type` is :class:`~ft.onto.base_ontology.Sentence`,
                then the features of each extractor will represent the
                information of a sentence. If this value is `None`, then all
                extractors will operate on the whole data pack.

            `"schemes"`: Dict
                A Dict containing the information about doing the
                pre-processing.
                The `key` is the tags provided by input `request`. The
                `value` is a `Dict` containing the information for doing
                pre-processing for that feature.

            `"schemes.tag.extractor"`:
                An instance of type
                :class:`~forte.data.extractor.BaseExtractor`.

            `"schemes.tag.converter"`:
                An instance of type :class:`~forte.data.converter.Converter`.

            `"schemes.tag.type"`: TrainPreprocessor.DATA_INPUT/DATA_OUTPUT
                Denoting whether this feature is the input or output feature.
        """
        if not self._request:
            self._parse_request(self._request)
        return self._request

    @property
    def device(self) -> device:
        r"""The device of the produced batches. For GPU training,
        set to current CUDA device.
        """
        return torch.device(self._config.preprocess.device)

    @property
    def config(self) -> Config:
        r"""A :class:`~forte.common.configuration.Config` maintaining all the
        configurable options for this `TrainPreprocessor`.
        """
        return self._config

    def get_train_batch_iterator(self) -> Iterator[Batch]:
        r"""
        This method mainly has four steps:

        1. Iterate over :class:`~forte.data.data_pack.DataPack`
           via pack iterator
        2. Extract :class:`~forte.data.converter.feature.Feature` from
           :class:`~forte.data.data_pack.DataPack`
        3. Batch :class:`~forte.data.converter.feature.Feature`
        4. (optional) Pad a batch of
           :class:`~forte.data.converter.feature.Feature`

        It will return an `iterator` of a batch of pre-processed data.

        Returns:
            An `Iterator` of type :class:`~texar.torch.data.Batch`

            Please refer to :meth:`collate` in
            :class:`~forte.data.data_pack_dataset.DataPackDataset` for details
            about its structure.
        """
        if not self._request:
            raise ValueError("Feature resource is not parsed")

        if not self._vocab_ready:
            raise ValueError("Vocab is not built")

        dataset_iter = self._build_dataset_iterator()

        return iter(dataset_iter)
