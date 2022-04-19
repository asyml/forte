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


from typing import (
    Dict,
    List,
    Iterable,
    Optional,
    Tuple,
    Type,
    Iterator,
    Any,
)


from forte.common.configuration import Config
from forte.data.base_pack import PackType
from forte.data.converter import Feature
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import Annotation
from forte.utils import get_class
from forte.data.batchers import ProcessingBatcher
from forte.common.exception import ValidationError


__all__ = [
    "FixedSizeDataPackBatcherWithExtractor",
]


# TODO: shouldn't implement a special extractor because we use extractors.
class FixedSizeDataPackBatcherWithExtractor(ProcessingBatcher):
    r"""This batcher uses extractor to extract features from
    dataset and group them into batch. In this class, more pools
    are added. One is `instance_pool`, which is used to record the
    instance from which feature is extracted. The other one is
    `feature_pool`, which is used to record features before they
    can be yield in batch.
    """

    def __init__(self):
        super().__init__()
        self._context_type: Type[EntryType] = None
        self._feature_scheme: Dict = {}
        self.batch_size: int = -1

        self.instance_pool: List[Annotation] = []
        self.feature_pool: List[Dict[str, Feature]] = []
        self.pool_size = 0
        self.batch_is_full = False

    def initialize(self, config: Config):
        super().initialize(config)

        if config["context_type"] is None:
            raise AttributeError("'context_type' cannot be None.")

        if config["batch_size"] is None:
            raise AttributeError("'batch_size' cannot be None.")

        if isinstance(config["context_type"], str):
            self._context_type = get_class(config["context_type"])
        else:
            self._context_type = config["context_type"]

        if not issubclass(self._context_type, Annotation):
            raise ValidationError(
                f"The provided context type {self._context_type} "
                f"is not an Annotation type."
            )

        self.batch_size = config["batch_size"]

        self.instance_pool.clear()
        self.feature_pool.clear()
        self.pool_size = 0
        self.batch_is_full = False

    def add_feature_scheme(self, tag: str, scheme: Dict[str, Any]):
        """
        Add feature scheme to the batcher.

        Args:
            tag (str): The name/tag of the scheme.
            scheme (str): The scheme content, which should be a dict
              containing the extractor and converter used to create features.
        """
        self._feature_scheme[tag] = scheme

    def collate(
        self, features_collection: List[Dict[str, Feature]]
    ) -> Dict[str, Dict[str, Any]]:
        r"""This function use the :class:`~forte.data.converter.converter.Converter`
        interface to turn a list of features into batches, where each feature
        is converted to tensor/matrix format. The resulting features are
        organized as a dictionary, where the keys are the feature names/tags,
        and the values are the converted features. Each feature contains the
        data and mask in `MatrixLike` form, as well as the original raw
        features.

        Args:
            features_collection: A list of features.

        Returns:
            A instance of `Dict[str, Union[Tensor, Dict]]`, which
            is a batch of features.
        """
        collections: Dict[str, List[Feature]] = {}
        for features in features_collection:
            for tag, feat in features.items():
                if tag not in collections:
                    collections[tag] = []
                collections[tag].append(feat)

        converted: Dict[str, Dict[str, Any]] = {}
        for tag, features in collections.items():
            converter = self._feature_scheme[tag]["converter"]
            data, masks = converter.convert(features)
            converted[tag] = {
                "data": data,
                "masks": masks,
                "features": features,
            }
        return converted

    def _should_yield(self) -> bool:
        return self.batch_is_full

    def flush(
        self,
    ) -> Iterator[
        Tuple[
            List[PackType],
            List[Optional[Annotation]],
            Dict[str, Dict[str, Any]],
        ]
    ]:
        r"""Flush data in batches. Each return value contains a tuple of 3
        items: the corresponding data pack, the list of annotation objects
        that represent the context type, and the features."""
        if self.pool_size > 0:
            yield (  # type: ignore
                self.data_pack_pool,
                self.instance_pool,
                self.collate(self.feature_pool),
            )
            self.data_pack_pool = []
            self.instance_pool = []
            self.feature_pool = []
            self.pool_size = 0

    def get_batch(
        self,
        input_pack: PackType,
    ) -> Iterator[Tuple[List[PackType], List[Optional[Annotation]], Dict]]:
        r"""By feeding data pack to this function, formatted features will
        be yielded based on the batching logic. Each element in the iterator is
        a triplet of datapack, context instance and batched data.

        Args:
            input_pack: The input data pack to get features from.

        Returns:
            An iterator of a tuple contains datapack, context instance and
            batch data.
        """
        # cache the new pack and generate batches

        for (batch, num) in self._get_data_batch(input_pack):
            self.data_pack_pool.extend(batch["packs"])
            self.instance_pool.extend(batch["contexts"])
            self.feature_pool.extend(batch["features"])
            self.pool_size += num

            # Yield a batch on two conditions.
            # 1. If we do not want to have batches from different pack, we
            # should yield since this pack is exhausted.
            # 2. We should also yield when the batcher condition is met:
            # i.e. ``_should_yield()`` is True.
            if not self._cross_pack or self._should_yield():
                yield from self.flush()  # type: ignore

    def _get_data_batch(
        self,
        data_pack: PackType,
    ) -> Iterable[Tuple[Dict[Any, Any], int]]:
        r"""Get data batches based on the requests.

        Args:
            data_pack: The data pack to retrieve data from.
        """
        packs: List[PackType] = []
        contexts: List[Annotation] = []
        features_collection: List[Dict[str, Feature]] = []
        current_size = self.pool_size

        for instance in data_pack.get(self._context_type):
            contexts.append(instance)
            features = {}
            for tag, scheme in self._feature_scheme.items():
                features[tag] = scheme["extractor"].extract(data_pack)
            packs.append(data_pack)
            features_collection.append(features)

            if len(contexts) == self.batch_size - current_size:
                self.batch_is_full = True

                batch = {
                    "packs": packs,
                    "contexts": contexts,
                    "features": features_collection,
                }

                yield batch, len(contexts)
                self.batch_is_full = False
                packs = []
                contexts = []
                features_collection = []
                current_size = self.pool_size

        # Flush the remaining data.
        if len(contexts) > 0:
            batch = {
                "packs": packs,
                "contexts": contexts,
                "features": features_collection,
            }
            yield batch, len(contexts)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Defines the configuration of this batcher, here:

            - context_type: The context scope to extract data from. It could be
              a annotation class or a string that is the fully qualified name
              of the annotation class.

            - feature_scheme: A dictionary of (extractor name, extractor) that
              can be used to extract features.

            - batch_size: The batch size, default is 10.

        Returns:
            The default configuration structure.

        """
        return {
            "context_type": None,
            "feature_scheme": None,
            "batch_size": 10,
        }
