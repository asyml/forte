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
from typing import Dict, Iterator, Type, Optional, List, Tuple, Union

import torch
from texar.torch import HParams
from texar.torch.data import IterDataSource, DatasetBase, Batch

from forte.data.converter.converter import Converter
from forte.data.converter.feature import Feature
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
    def __init__(self,
                 data_source: DataPackDataSource,
                 feature_schemes: Dict,
                 hparams: Union[Dict, HParams] = None,
                 device: Optional[torch.device] = None):
        self._data_source = data_source
        self._feature_scheme = feature_schemes

        super().__init__(self._data_source, hparams, device)

    def process(self, raw_example: RawExample) -> FeatureCollection:
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
        batch_size = len(examples)

        example_collection: Dict[str, List] = {}
        for example in examples:
            for tag, feature in example.items():
                if tag not in example_collection:
                    example_collection[tag] = []
                example_collection[tag].append(feature)

        tensor_collection: Dict[str, Dict[str, torch.Tensor]] = {}
        for tag, features in example_collection.items():
            need_pad: bool = self._feature_scheme[tag]["need_pad"]

            if tag not in tensor_collection:
                tensor_collection[tag] = {}

            if need_pad:
                converter: Converter = \
                    self._feature_scheme[tag]["converter"]
                tensor, mask = converter.convert(features)
                tensor_collection[tag]["tensor"] = tensor
                tensor_collection[tag]["mask"] = mask

            tensor_collection[tag]["features"] = features

        return Batch(batch_size, **tensor_collection)
