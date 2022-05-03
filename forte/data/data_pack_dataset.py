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
Provide data across multiple data packs during training. A data pack iterator
iterates over each single data example across multiple data packs. A data pack
data set represents the dataset of a bunch of data packs. A raw example
represents a single data point in the dataset. A feature collection represents
an extracted feature corresponding to an input data point.
"""
from typing import Dict, Iterator, Type, Optional, List, Tuple, Union, Any

from asyml_utilities.hyperparams import HParams


from forte.data.converter import Converter
from forte.data.converter import Feature
from forte.data.data_pack import DataPack
from forte.data.base_extractor import BaseExtractor
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import Annotation
from forte.data.types import DataRequest
from forte.utils import create_import_error_msg

try:
    import torch
except ImportError as e:
    raise ImportError(
        create_import_error_msg("torch", "extractor", "data pack dataset")
    ) from e

try:
    from texar.torch.data import IterDataSource, DatasetBase, Batch
except ImportError as e:
    raise ImportError(
        create_import_error_msg(
            "texar-pytorch", "extractor", "data pack dataset"
        )
    ) from e


__all__ = [
    "DataPackIterator",
    "DataPackDataset",
    "RawExample",
    "FeatureCollection",
]

# An instance is a single data point from data pack
RawExample = Tuple[int, DataPack]
FeatureCollection = Dict[str, Feature]


class DataPackIterator(IterDataSource):
    """
    An iterator generating data example from a stream of data packs.

    Args:
        pack_iterator: An iterator of
            :class:`~forte.data.data_pack.DataPack`.
        context_type: The granularity of a single example which
            could be any :class:`~forte.data.ontology.top.Annotation` type. For example, it can be
            :class:`~ft.onto.base_ontology.Sentence`, then each training example
            will represent the information of a sentence.
        request: The request of type `Dict` sent to
            :class:`~forte.data.data_pack.DataPack` to query
            specific data.
        skip_k: Will skip the first `skip_k` instances and generate
            data from the (`skip_k` + 1)th instance.

    Returns:
        An `Iterator` that each time produces a `Tuple` of an `tid`
        (of type `int`) and a data pack
        (of type :class:`~forte.data.data_pack.DataPack`).

    Here is an example usage:

        .. code-block:: python

            file_path: str = "data_samples/data_pack_dataset_test"
            reader = CoNLL03Reader()
            context_type = Sentence
            request = {Sentence: []}
            skip_k = 0

            train_pl: Pipeline = Pipeline()
            train_pl.set_reader(reader)
            train_pl.initialize()
            pack_iterator: Iterator[PackType] =
                train_pl.process_dataset(file_path)

            iterator: DataPackIterator = DataPackIterator(pack_iterator,
                                                          context_type,
                                                          request,
                                                          skip_k)

            for tid, data_pack in iterator:
                # process tid and data_pack

    .. note::
        For parameters `context_type`, `request`, `skip_k`, please refer to
        :meth:`~forte.data.data_pack.DataPack.get_data` in :class:`~forte.data.data_pack.DataPack`.
    """

    def __init__(
        self,
        pack_iterator: Iterator[DataPack],
        context_type: Type[Annotation],
        request: Optional[DataRequest] = None,
        skip_k: int = 0,
    ):
        super().__init__(self)

        self._get_data_args: Dict = {
            "context_type": context_type,
            "request": request,
            "skip_k": skip_k,
        }

        self._data_pack_iter: Iterator[DataPack] = pack_iterator
        self._instance_iter: Optional[Iterator[Dict[str, Any]]] = None
        self._curr_data_pack: Optional[DataPack] = None

    def __iter__(self):
        return self

    def __next__(self) -> RawExample:
        if self._curr_data_pack is None:
            self._curr_data_pack = next(self._data_pack_iter)
            self._instance_iter = self._curr_data_pack.get_data(
                **self._get_data_args
            )

        if self._instance_iter is None:
            raise ValueError("Instance iterator is None")

        try:
            return next(self._instance_iter)["tid"], self._curr_data_pack
        except StopIteration:
            # Current data pack has no more instance. Go to next data pack.
            self._curr_data_pack = next(self._data_pack_iter)
            self._instance_iter = self._curr_data_pack.get_data(
                **self._get_data_args
            )

        return next(self._instance_iter)["tid"], self._curr_data_pack


class DataPackDataset(DatasetBase):
    """
    A dataset representing data packs. Calling an
    :class:`~texar.torch.data.DataIterator`
    over this `DataPackDataset` will produce an `Iterate` over batch of examples
    parsed by a reader from given data packs.

    Args:
        data_source: A data source of type
            :class:`~forte.data.data_pack_dataset.DataPackIterator`.
        feature_schemes: A `Dict` containing all the information to do
            data pre-processing. This is exactly the same as the `schemes` in
            :meth:`~forte.train_preprocessor.TrainPreprocessor.request`.
        hparams: A `dict` or instance of
            :class:`~texar.torch.HParams` containing
            hyperparameters. See
            :meth:`~texar.torch.data.DatasetBase.default_hparams` in
            :class:`~texar.torch.data.DatasetBase` for the defaults.
        device: The device of the produced batches. For GPU training,
            set to current CUDA device.
    """

    def __init__(
        self,
        data_source: DataPackIterator,
        feature_schemes: Dict,
        hparams: Union[Dict, HParams] = None,
        device: Optional[torch.device] = None,
    ):
        self._data_source: DataPackIterator = data_source
        self._feature_scheme: Dict = feature_schemes

        super().__init__(self._data_source, hparams, device)

    def process(self, raw_example: RawExample) -> FeatureCollection:
        """
        Given an input which is a single data example, extract feature from it.

        Args:
            raw_example (tuple(dict, DataPack)): A `Tuple` where

              - The first element is a `Dict` produced by
                :meth:`~forte.data.data_pack.DataPack.get_data` in
                :class:`~forte.data.data_pack.DataPack`.

              - The second element is an instance of type
                :class:`~forte.data.data_pack.DataPack`.

        Returns:
            A `Dict` mapping from user-specified tags to the
            :class:`~forte.data.converter.Feature` extracted.

            .. note::
                Please refer to
                :meth:`~forte.train_preprocessor.TrainPreprocessor.request` for
                details about user-specified tags.
        """
        tid: int = raw_example[0]
        data_pack: DataPack = raw_example[1]
        instance_entry: EntryType = data_pack.get_entry(tid)  # type:ignore
        feature_collection: FeatureCollection = {}

        for tag, scheme in self._feature_scheme.items():
            extractor: BaseExtractor = scheme["extractor"]
            feature: Feature = extractor.extract(data_pack, instance_entry)
            feature_collection[tag] = feature

        return feature_collection

    def collate(self, examples: List[FeatureCollection]) -> Batch:
        """
        Given a batch of output from :meth:`process`, produce pre-processed
        data as well as masks and features.

        Args:
            examples: A `List` of result from :meth:`process`.

        Returns:
            A Texar Pytorch :class:`~Texar.torch.data.Batch`
            It can be treated as a `Dict` with the following structure:



            - `"data"`: List or `np.ndarray` or `torch.tensor`
              The pre-processed data.

              Please refer to :class:`~forte.data.converter.converter.Converter` for
              details.

            - `"masks"`: `np.ndarray` or `torch.tensor`
              All the masks for pre-processed data.

              Please refer to :class:`~forte.data.converter.converter.Converter` for
              details.

            - `"features"`: List[Feature]
              A List of :class:`~forte.data.converter.feature.Feature`. This
              is useful when users want to do customized pre-processing.

              Please refer to :class:`~forte.data.converter.Feature` for
              details.

            .. code-block:: python

                {
                    "tag_a": {
                        "data": <tensor>,
                        "masks": [<tensor1>, <tensor2>, ...],
                        "features": [<feature1>, <feature2>, ...]
                    },
                    "tag_b": {
                        "data": Tensor,
                        "masks": [<tensor1>, <tensor2>, ...],
                        "features": [<feature1>, <feature2>, ...]
                    }
                }


            .. note::
                The first level key in returned `batch` is the user-specified
                tags. Please refer to
                :meth:`~forte.train_preprocessor.TrainPreprocessor.request` for
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
            if tag not in tensor_collection:
                tensor_collection[tag] = {}

            converter: Converter = self._feature_scheme[tag]["converter"]
            data, masks = converter.convert(features)
            tensor_collection[tag]["data"] = data
            tensor_collection[tag]["masks"] = masks
            tensor_collection[tag]["features"] = features

        return Batch(batch_size, **tensor_collection)
