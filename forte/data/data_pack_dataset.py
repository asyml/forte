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
from typing import Dict, Iterator, Type, Optional, List, Tuple, Union, Any

import torch
from texar.torch import HParams
from texar.torch.data import IterDataSource, DatasetBase, Batch

from forte.data.converter import Converter
from forte.data.converter import Feature
from forte.data.data_pack import DataPack
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import Annotation
from forte.data.readers.base_reader import PackReader
from forte.data.types import DataRequest

# An instance is a single data point from data pack
Instance = Dict
RawExample = Tuple[Instance, DataPack]
FeatureCollection = Dict[str, Feature]


class DataPackIterator:
    """
    An iterator over single data example from multiple data packs.

    Args:
        reader (PackReader): A reader of
            :class:`forte.data.readers.base_reader.PackReader` that will
            parse dataset files into data packs.
        pack_dir (str): A string of the directory path that includes all the
            dataset files to be parsed.
        context_type: The granularity of a single example which
            could be any ``Annotation`` type. For example, it can be
            :class:`ft.onto.base_ontology.Sentence`, then each training example
            will represent the information of a sentence.
        request: The request of type `Dict` sent to
            :class:`forte.data.readers.base_reader.PackReader` to query
            specific data.
        skip_k (int): Will skip the first `skip_k` instances and generate
            data from the (`offset` + 1)th instance.

    .. note::
        For parameters `context_type`, `request`, `skip_k`, please refer to
        :meth:`get_data()` in :class:`forte.data.data_pack.DataPack`.
    """
    def __init__(self,
                 reader: PackReader,
                 pack_dir: str,
                 context_type: Type[Annotation],
                 request: Optional[DataRequest] = None,
                 skip_k: int = 0):
        self._reader: PackReader = reader
        self._pack_dir: str = pack_dir
        self._get_data_args: Dict = {
            "context_type": context_type,
            "request": request,
            "skip_k": skip_k
        }

        self._data_pack_iter: Iterator[DataPack] = \
            self._reader.iter(self._pack_dir)
        self._instance_iter: Optional[Iterator[Instance]] = None
        self._curr_data_pack: Optional[DataPack] = None

    def __iter__(self):
        return self

    def __next__(self) -> RawExample:
        try:
            if self._curr_data_pack is None:
                self._curr_data_pack = next(self._data_pack_iter)
                self._instance_iter = \
                    self._curr_data_pack.get_data(**self._get_data_args)

            assert self._instance_iter is not None

            try:
                return next(self._instance_iter), self._curr_data_pack
            except StopIteration:
                # Current data pack has no more instance. Go to next data pack.
                self._curr_data_pack = next(self._data_pack_iter)
                self._instance_iter = \
                    self._curr_data_pack.get_data(**self._get_data_args)

            return next(self._instance_iter), self._curr_data_pack
        except StopIteration:
            raise StopIteration()


class DataPackDataSource(IterDataSource):
    """
    A data source consists of data packs. It contains an iterator over
    :class:`forte.data.data_pack_dataset.DataPackIterator`.

    Args:
        reader (PackReader): A reader of
            :class:`forte.data.readers.base_reader.PackReader` that will
            parse dataset files into data packs.
        pack_dir (str): A string of the directory path that includes all the
            dataset files to be parsed.
        context_type: The granularity of a single example which
            could be any ``Annotation`` type. For example, it can be
            :class:`ft.onto.base_ontology.Sentence`, then each training example
            will represent the information of a sentence.
        request: The request of type `Dict` sent to
            :class:`forte.data.readers.base_reader.PackReader` to query
            specific data.
        skip_k (int): Will skip the first `skip_k` instances and generate
            data from the (`offset` + 1)th instance.

    .. note::
        For parameters `context_type`, `request`, `skip_k`, please refer to
        :meth:`get_data()` in :class:`forte.data.data_pack.DataPack`.
    """
    def __init__(self,
                 reader: PackReader,
                 pack_dir: str,
                 context_type: Type[Annotation],
                 request: Optional[DataRequest] = None,
                 skip_k: int = 0):
        self._iterator: Iterator = DataPackIterator(reader,
                                                    pack_dir,
                                                    context_type,
                                                    request,
                                                    skip_k)
        super().__init__(self)

    def __iter__(self):
        return self._iterator


class DataPackDataset(DatasetBase):
    """
    A dataset representing data packs. Calling an
    `DataIterator
    <https://texar-pytorch.readthedocs.io/en/latest/code/data.html#dataiterator>`_
    over this `DataPackDataset` will produce an `Iterate` over batch of examples
    parsed by a reader from given data packs.

    Args:
        data_source: A data source of type
            :class:`forte.data.data_pack_dataset.DataPackDataSource`.
        feature_schemes (dict): A `Dict` containing all the information to do
            data pre-processing. This is exactly the same as the `schemes` in
            `feature_resource`. Please refer to :meth:`feature_resource` in
            :class:`forte.train_preprocessor.TrainPreprocessor` for details.
        hparams: A `dict` or instance of :
            class:`forte.common.configuration.Config` containing
            hyperparameters. See :meth:`default_hparams` in
            `DatasetBase
            <https://texar-pytorch.readthedocs.io/en/latest/code/data.html#datasetbase>`
            for the defaults.
        device: The device of the produced batches. For GPU training,
            set to current CUDA device.
    """
    def __init__(self,
                 data_source: DataPackDataSource,
                 feature_schemes: Dict,
                 hparams: Union[Dict, HParams] = None,
                 device: Optional[torch.device] = None):
        self._data_source = data_source
        self._feature_scheme = feature_schemes

        super().__init__(self._data_source, hparams, device)

    def process(self, raw_example: RawExample) -> FeatureCollection:
        """
        Given an input which is a single data example, extract feature from it.

        Args:
            raw_example (tuple(dict, DataPack)): A `Tuple` where

                The first element is a `Dict` produced by :meth:`get_data()` in
                :class:`forte.data.data_pack.DataPack`.

                The second element is an instance of type
                :class:`forte.data.data_pack.DataPack`.

        Returns:
            A `Dict` mapping from user-specified tags to the
            :class:`forte.data.converter.Feature` extracted.

            .. note::
                Please refer to Please refer to :meth:`feature_resource` in
                :class:`forte.train_preprocessor.TrainPreprocessor` for details
                about user-specified tags.
        """
        instance: Instance = raw_example[0]
        data_pack: DataPack = raw_example[1]
        instance_entry: EntryType = data_pack.get_entry(  # type: ignore
            instance["tid"])
        feature_collection: FeatureCollection = {}

        for tag, scheme in self._feature_scheme.items():
            extractor: BaseExtractor = scheme["extractor"]
            feature: Feature = extractor.extract(data_pack, instance_entry)
            feature_collection[tag] = feature

        return feature_collection

    def collate(self, examples: List[FeatureCollection]) -> Batch:
        """
        Given a batch of output from :meth:`process`, produce a `Tensor`
        containing pre-processed data including tensors, masks and features.

        Args:
            examples: A `List` of result from :meth:`process`.

        Returns:
            A texar `Batch
            <https://texar-pytorch.readthedocs.io/en/latest/code/data.html#batch>`_.
            It can be treated as a `Dict` with the following structure:

            .. code-block:: python

                {
                    "tag_a": {
                        "tensor": <tensor>,
                        "masks": [<tensor1>, <tensor2>, ...],
                        "features": [<feature1>, <feature2>, ...]
                    },
                    "tag_b": {
                        "tensor": Tensor,
                        "masks": [<tensor1>, <tensor2>, ...],
                        "features": [<feature1>, <feature2>, ...]
                    }
                }

            `"tensor"`: Tensor
                The tensor representing the Feature. This should be sent to
                model for training.

                If the option `need_pad` in input`request` is False, this is
                not given.

            `"masks"`: List[Tensor]
                The `mask` is actually a list of tensors where each tensor
                representing the mask for that dimension. For example, mask[i]
                is a tensor representing the mask for dimension i.

                If the option `need_pad` in input`request` is False, this is
                not given.

            `"features"`: List[Feature]
                A List of :class:`forte.data.converter.feature.Feature`. This is
                useful when users want to do customized pre-processing.

            .. note::
                The first level key in returned `batch` is the user-specified
                tags. Please refer to :meth:`feature_resource`
                in :class:`forte.train_preprocessor.TrainPreprocessor` for
                details about user-specified tags.
        """
        batch_size = len(examples)

        example_collection: Dict[str, List] = {}
        for example in examples:
            for tag, feature in example.items():
                if tag not in example_collection:
                    example_collection[tag] = []
                example_collection[tag].append(feature)

        tensor_collection: Dict[str, Dict[str, Any]] = {}
        for tag, features in example_collection.items():
            need_pad: bool = self._feature_scheme[tag]["need_pad"]

            if tag not in tensor_collection:
                tensor_collection[tag] = {}

            if need_pad:
                converter: Converter = \
                    self._feature_scheme[tag]["converter"]
                tensor, mask = converter.convert(features)
                tensor_collection[tag]["tensor"] = tensor
                tensor_collection[tag]["masks"] = mask

            tensor_collection[tag]["features"] = features

        return Batch(batch_size, **tensor_collection)
