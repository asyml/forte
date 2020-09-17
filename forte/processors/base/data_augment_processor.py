# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Processors that augment the data.
"""
from typing import Iterable, Tuple
from abc import abstractmethod
from forte.data.caster import Caster
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack


__all__ = [
    "BaseDataAugmentProcessor"
]


class BaseDataAugmentProcessor(Caster[DataPack, MultiPack]):
    r"""The base class of processors that augment data.
    This processor will call :func:'augment_algo' to get the
    augmented texts which are semantically similar to the original texts.

    With the augmented texts, :func:'build_pack' builds the data packs by either
        (1) Copying the original data pack and modifying it, or
        (2) Building a new data pack from the augmented texts.
    The implementation of :func:'build_pack' varies for different types of datasets.
    For example, option (2) is available for datasets like sentiment analysis
    where we can rebuild the annotations easily. But for treebanks where structure
    information is stored in the original data pack, we can only rely on the option(1).
    """
    def __init__(self):
        super().__init__()

    def cast(self, pack: DataPack) -> MultiPack:
        """
        Augment the data-pack into a multi-pack.

        Args:
            pack: The data pack to be augmented

        Returns: An iterator that produces the augmented multi pack.

        """
        p = MultiPack()
        p.add_pack_(pack, "original")
        augmented_texts: Iterable[Tuple[str, str]] = self.augment_algo(pack)
        for pack_name, text in augmented_texts:
            new_pack: DataPack = self.build_pack(pack, text)
            p.add_pack_(new_pack, pack_name)
        return p

    @abstractmethod
    def augment_algo(self, pack: DataPack) -> Iterable[Tuple[str, str]]:
        r"""The method that augments the input data pack.
        Different algorithms can be inserted here,
        such as word replacement or back-translation.
        The outputs are raw strings, instead of Forte data structures.

        Args:
            pack: The data pack to be augmented.

        Returns: An iterator of tuples [pack_name, augmented_text].
        """
        raise NotImplementedError

    @abstractmethod
    def build_pack(self, original_pack: DataPack, augmented_text: str):
        r"""The method that builds a data pack
        from both original data pack and augmented text.

        For data pack where only annotations like Token, Sentence and Document
        are included, it is simple to build a data pack only from the augmented text.

        However, for datasets like treebanks, we need the tree structure information
        embedded in the original data pack. We cannot infer that from the augmented text.
        Instead of building a new data pack, We can only copy the original one and modify
        it based on the augmented text. This will impose some constraints on the augmentation
        algorithm, for example, only replacement-based methods can be used for treebanks
        to preserve its original structure.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for AllenNLP.
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - pack_builder_type: defines how :func:'build_pack' build new packs from
            original data pack and augmented texts, e.g., it will build the annotations
            only from the augmented text when pack_builder_type="plainText".
            - augment_ontologies: defines what are the ontologies to build in
            :func:'build_pack'. This should align with the "pack_builder_type".
            For example, when the pack_builder_type="plainText", the augment_ontologies
            must not include Link or ConstituencyNode.
        """
        config = super().default_configs()
        config.update({
            'pack_builder_type': "plainText",
            'augment_ontologies': ["Token", "Sentence", "Document"]
        })
        return config