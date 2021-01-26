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
from texar.torch.data import DataIterator, Batch
import torch
from torch import device

from forte.common.configuration import Config
from forte.data.converter import Converter
from forte.data.data_pack import DataPack
from forte.data.data_pack_dataset import DataPackDataset, DataPackIterator
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType

logger = logging.getLogger(__name__)

__all__ = [
    "TrainPreprocessor"
]


class TrainPreprocessor:
    r"""
    `TrainPreprocessor` provides the functionality of doing pre-processing work
    including building vocabulary, extracting the features, batching and
    padding (optional). The main functionality is provided by its method
    :meth:`get_train_batch_iterator` which will return an `iterator` over the
    batch of preprocessed data. Please refer to the documentation of
    that method for how the pre-processing is done.

    `TrainPreprocessor` will maintain a Config that stores all the configurable
    parameters for various components.

    `TrainPreprocessor` will also accept a user request. Internally it will
    parse this user request and store the parsed result.

    Args:
        pack_iterator (Iterator[DataPack]): An iterator of
            :class:`~forte.data.data_pack.DataPack`.
        request (Dict): A request that specifies how to do train pre-processing.
            Please refer to :meth:`request` for details.
        config: A `Dict` or :class:`~forte.common.configuration.Config` that
            configs this preprocessor. See :meth:`default_configs` for
            the defaults.


    .. note::
        For parameters `request`, user does not necessarily need to provide
        `converter`. If no `converter` is specified, a default converter of
        type :class:`~forte.data.converter.Converter` will be picked.
    """

    DATA_INPUT = 0
    DATA_OUTPUT = 1

    def __init__(self,
                 pack_iterator: Iterator[DataPack],
                 request: Dict,
                 config: Optional[Union[Config, Dict]] = None):
        self._config: Config = \
            Config(config, default_hparams=self.default_configs())
        self._validate_config()

        self._pack_iterator: Iterator[DataPack] = pack_iterator
        self._cached_packs: List[DataPack] = []

        self._user_request: Dict = request
        self._request: Dict = {}
        self._request_ready: bool = False
        self._vocab_ready: bool = False

        self._parse_request(self._user_request)
        self._build_vocab()

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
            "dataset": DataPackDataset.default_hparams()
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

        assert "scope" in request, \
            "Field not found for data request: `scope`"
        assert "schemes" in request, \
            "Field not found for data request: `schemes`"

        resource_schemes: Dict[str, Dict] = {}
        # Used for check dependency between different extractors
        scheme_group: Dict[str, Dict] = {
            "dependent": {}, "dependee": {}
        }

        for tag, scheme in request["schemes"].items():
            assert "extractor" in scheme, \
                "Field not found for data request scheme: `extractor`"
            assert "type" in scheme, \
                "Field not found for data request scheme: `type`"
            resource_schemes[tag] = {}

            if not isinstance(scheme["extractor"], BaseExtractor):
                raise RuntimeError("Invalid extractor: ", scheme["extractor"])

            extractor: BaseExtractor = scheme["extractor"]

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

            # Create default converter if there is no given converter
            if "converter" not in scheme:
                converter: Converter = Converter({})
                scheme["converter"] = converter

        # Check dependency
        for _, dependent_extractors in scheme_group["dependent"].items():
            for dependent_extractor in dependent_extractors:
                based_on: Entry = dependent_extractor.based_on
                if based_on not in scheme_group["dependee"]:
                    raise ValueError(
                        "Extractor {} needs the entry {} to do extraction "
                        "processing but it is not extracted by any other "
                        "extractors given in request".
                            format(based_on, dependent_extractor.tag))

        self._request = request
        self._request_ready = True

    def _build_vocab(self):
        scope: EntryType = self._request["scope"]
        schemes: Dict = self._request["schemes"]

        # TODO: clear vocab?

        # Cached all data packs
        for data_pack in self._pack_iterator:
            self._cached_packs.append(data_pack)

        for _, scheme in schemes.items():
            extractor: BaseExtractor = scheme["extractor"]
            if extractor.vocab_method != "raw":
                for data_pack in self._cached_packs:
                    for instance in data_pack.get(scope):
                        extractor.update_vocab(data_pack, instance)

        self._vocab_ready = True

    def _build_dataset_iterator(self) \
            -> DataIterator:
        scope: Type[EntryType] = self._request["scope"]  # type: ignore
        schemes: Dict[str, Dict[str, Any]] = self._request["schemes"]

        data_source = DataPackIterator(pack_iterator=iter(self._cached_packs),
                                       context_type=scope,
                                       request={scope: []})

        dataset = DataPackDataset(data_source,
                                  schemes,
                                  self._config.dataset,
                                  self.device)
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
                "scope": ft.onto.Sentence
                "schemes": {
                    "text_tag": {
                        "extractor": forte.data.extractor.AttributeExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_INPUT,
                    },
                    "char_tag" {
                        "extractor": forte.data.extractor.CharExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_INPUT,
                    }
                    "ner_tag": {
                        "extractor":
                            forte.data.extractor.BioSeqTaggingExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_OUTPUT,
                    }
                }
            }

    Here:

        `"scope"`: Entry
            A class of type :class:`~forte.data.ontology.core.Entry` The
            granularity to separate data into different examples. For example,
            if `scope` is :class:`~ft.onto.base_ontology.Sentence`, then each
            training example will represent the information of a sentence.

        `"schemes"`: `Dict`
            A `Dict` containing the information about doing the pre-processing.
            The `key` is the tags provided by input `request`. The `value` is a
            `Dict` containing the information for doing pre-processing for that
            feature.

        `"schemes.tag.extractor"`: Extractor
            An instance of type :class:`~forte.data.extractor.BaseExtractor`.

        `"schemes.tag.converter"`: Converter
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

        It will return an `iterator` of a batch of preprocessed data.

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
