# ***automatically_generated***
# ***source json:../testing/sample_onto/sample_ndarray_attribute.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology NdArrayEntry. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import FNdArray
from forte.data.ontology.top import Generics

__all__ = [
    "NdEntry1",
    "NdEntry2",
    "NdEntry3",
]


@dataclass
class NdEntry1(Generics):
    """
    A N-dimensional entity, having dtype and shape specified.
    Attributes:
        value (FNdArray):
    """

    value: FNdArray

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: FNdArray = FNdArray(shape=[2, 2], dtype='int')


@dataclass
class NdEntry2(Generics):
    """
    A N-dimensional entity. Only dtype is specified
    Attributes:
        value (FNdArray):
    """

    value: FNdArray

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: FNdArray = FNdArray(shape=None, dtype='int')


@dataclass
class NdEntry3(Generics):
    """
    A N-dimensional entity. Only shape is specified
    Attributes:
        value (FNdArray):
    """

    value: FNdArray

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: FNdArray = FNdArray(shape=[2, 2], dtype=None)
