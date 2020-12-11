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
from texar.torch.data import DataIterator, Batch
import torch
from torch import device

from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.data.converter import Converter
from forte.data.data_pack_dataset import DataPackDataSource, \
    DataPackDataset
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType
from forte.data.readers.base_reader import PackReader

logger = logging.getLogger(__name__)


class TrainPreprocessor:
    r"""
    `TrainPreprocessor` provides the functionality of doing pre-processing work
    including building vocabulary, extracting the features, batching and
    padding (optional). The main functionality is provided by its method
    :meth:`get_train_batch_iterator` which will return an `iterator` over the
    batch of preprocessed data. Please refer to the documentation of
    that method for how the preprocessing is done.

    `TrainPreprocessor` will maintain a Config that stores all the configurable
    parameters for various components.

    `TrainPreprocessor` will also accept a user request. Internally it will
    parse this user request and store the parsed result as a dictionary
    into `feature_resource`.

    Args:
        train_reader (PackReader): An object of class
            :class:`forte.data.readers.PackReader` that parses given
            dataset files.
        request (Dict): A request that specifies how to do train pre-processing.
            See below for details. An example is given below.
        config: A `Dict` or :class:`forte.common.configuration.Config` that
            configs this preprocessor. See :meth:`default_configs` for
            the defaults.
        reader_config: A `Dict` or :class:`forte.common.configuration.Config`
            that configs the given `train_reader`. See `default_configs` for
            the specific reader for defaults.

    Here is the detailed explanation for `request`:

    `"scope"`: Entry
        A class of type :class:`forte.data.ontology.core.Entry` The granularity
        to separate data into different examples. For example, if `scope` is
        :class:`ft.onto.base_ontology.Sentence`, then each training example
        will represent the information of a sentence.

    `"schemes"`: Dict
        The directory containing pre-processing information. It will guide the
        `TrainPreprocessor` how to do pre-processing. The key will be
        user-defined tags that denotes a type of Feature. The value will be
        various options for how to pre-processing that Feature. Lots of the
        options are for extracting process and those are all the options
        supported by various :mod:`forte.data.extractor`. Please refer to
        those documentations for detailed configuration.

    `"schemes.tag.need_pad"`: bool
        Whether `TrainPreprocessor` need to do padding after extracting data
        into :class:`forte.data.converter.feature.Feature`. Default is True.

    Example request

        .. code-block:: python

            data_request = {
                "scope": Sentence,
                "schemes": {
                    "text_tag": {
                        "entry_type": ft.onto.base_ontology.Token,
                        "vocab_method": "indexing",
                        "attribute_get": "text",
                        "type": TrainPreprocessor.DATA_OUTPUT,
                        "extractor": forte.data.extractor.AttributeExtractor
                    },
                    "char_tag": {
                        "entry_type": ft.onto.base_ontology.Token,
                        "vocab_method": "indexing",
                        "max_char_length": config.config_data.max_char_length,
                        "type": TrainPreprocessor.DATA_OUTPUT,
                        "extractor": forte.data.extractor.CharExtractor
                    },
                    "ner_tag": {
                        "entry_type": ft.onto.base_ontology.EntityMention,
                        "attribute": "ner_type",
                        "based_on": Token,
                        "vocab_method": "indexing",
                        "type": TrainPreprocessor.DATA_OUTPUT,
                        "extractor": forte.data.extractor.BioSeqTaggingExtractor
                    }
                }
            }
    """

    DATA_INPUT = 0
    DATA_OUTPUT = 1

    def __init__(self,
                 train_reader: PackReader,
                 request: Dict,
                 config: Optional[Union[Config, Dict]] = None,
                 reader_config: Optional[Union[Config, Dict]] = None):
        self._config: Config = \
            Config(config, default_hparams=self.default_configs())
        self._validate_config()

        self._train_reader: PackReader = train_reader

        self._user_request: Dict = request
        self._feature_resource: Dict = {}
        self._feature_resource_ready: bool = False
        self._vocab_ready: bool = False

        # Initialize reader
        self._train_reader.initialize(Resources(),
                                      self._train_reader.make_configs(
                                          configs=reader_config))

        if not self._config.preprocess.lazy_parse_request:
            self._parse_request(self._user_request)

        if not self._config.preprocess.lazy_build_vocab:
            assert not self._config.preprocess.lazy_parse_request
            self._build_vocab()

    @staticmethod
    def default_configs():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "preprocess": {
                            "pack_dir": "",
                            "lazy_parse_request": False,
                            "lazy_build_vocab": False,
                            "device": "cpu",
                },
                "dataset": DataPackDataset.default_hparams()
            }

        Here:

        `"preprocess.pack_dir"`: str
            The directory containing all the training dataset files

            .. note::
                Different reader will require different file extensions. Please
                refer to the specific reader for the file extension.

        `"preprocessor.lazy_parse_request"`: bool
            If False (default), preprocessor will parse input user request when
            the `TrainPreprocessor` instance is created. If False, it will parse
            user request when :meth:`get_train_batch_iterator` is called.

        `"preprocessor.lazy_build_vocab"`: bool
            If False (default), preprocessor will iterate over all dataset files
            and build the vocabulary when the `TrainPreprocessor` instance is
            created. If False, it will build the vocabulary when
            :meth:`get_train_batch_iterator` is called.

        `"preprocessor.device"`:
            The device of the produced batches. For GPU training,
            set to current CUDA device.

        `"dataset"`:
            This contains all the configurable options same as
            :class:`forte.data.data_pack_dataset.DataPackDataset`.

        """

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
        This method has two responsibilities:
        1. parse the given data request and stored internally into resource
        2. validate if the given data request is valid
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

            except Exception as e:
                logger.error("Error instantiate extractor: %s", str(e))
                raise

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
                    if extractor.vocab_method != "raw":
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
        # pylint: disable=line-too-long
        r"""A `Dict` containing all the information needed for doing the
        pre-processing. This is obtained via parsing the input `request`

        An example `feature_resource` is:

        .. code-block:: python

            feature_resource = {
                "scope": ft.onto.Sentence
                "schemes": {
                    "text_tag": {
                        "extractor": forte.data.extractor.AttributeExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_INPUT,
                        "need_pad": True
                    },
                    "char_tag" {
                        "extractor": forte.data.extractor.CharExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_INPUT,
                        "need_pad": True
                    }
                    "ner_tag": {
                        "extractor": forte.data.extractor.BioSeqTaggingExtractor,
                        "converter": forte.data.converter.Converter,
                        "type": TrainPreprocessor.DATA_OUTPUT,
                        "need_pad": True
                    }
                }
            }

        Here:

        `"scope"`: Entry
            Same as the `scope` provided by input `request`.

        `"schemes"`: Dict
            A `Dict` containing the information about doing the pre-processing.
            The `key` is the tags provided by input `request`. The `value` is a
            `Dict` containing the information for doing pre-processing for that
            feature.

        `"schemes.tag.extractor"`: Extractor
            An instance of type :class:`forte.data.extractor.BaseExtractor`.

        `"schemes.tag.converter"`: Converter
            An instance of type :class:`forte.data.converter.Converter`.

        `"schemes.tag.type"`: TrainPreprocessor.DATA_INPUT/DATA_OUTPUT
            Denoting whether this feature is the input or output feature.

        `"schemes.tag.need_pad"`: bool
            Whether the padding need to be done for this feature.

        """
        if not self._feature_resource_ready:
            self._parse_request(self._user_request)
        return self._feature_resource

    @property
    def user_request(self) -> Dict:
        r"""A `Dict` passed by users when
        :class:`forte.train_preprocessor.TrainPreprocessor` instance is created.
        Please refer to class documentation for the detailed format.
        """
        return self._user_request

    @property
    def device(self) -> device:
        r"""The device of the produced batches. For GPU training,
        set to current CUDA device.
        """
        return torch.device(self._config.preprocess.device)

    @property
    def config(self) -> Config:
        r"""A :class:`forte.common.configuration.Config` maintaining all the
        configurable options for this `TrainPreprocessor`.
        """
        return self._config

    @property
    def state(self) -> Dict:
        r"""A `Dict` maintaining all the serializable states for this
        `TrainPreprocessor`. This is typically used to save the training state.
        """
        return {
            "feature_resource": self._feature_resource,
            "user_request": self.user_request,
            "configs": self._config
        }

    def save_state(self, filename: str):
        r"""It will serialize `TrainPreprocessor` state into a disk file given
        by `filename`.

        Args:
            filename (str): the file to save the state.
        """
        torch.save(self.state, filename)

    def get_train_batch_iterator(self) -> Iterator[Batch]:
        r"""
        This method mainly has four steps:

        1. Parse dataset file into :class:`forte.data.data_pack.DataPack`
           via train reader
        2. Extract :class:`forte.data.converter.feature.Feature` from
           :class:`forte.data.data_pack.DataPack`
        3. Batch :class:`forte.data.converter.feature.Feature`
        4. (optional) Pad a batch of
           :class:`forte.data.converter.feature.Feature`

        It will return an `iterator` of a batch of preprocessed data.

        Returns:
            An `Iterator` of the `Batch
            <https://texar-pytorch.readthedocs.io/en/latest/code/data.html#batch>`_.

            Please refer to :meth:`collate` in
            :class:`forte.data.data_pack_dataset.DataPackDataset` for details
            about its structure.
        """
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
