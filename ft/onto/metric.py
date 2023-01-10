# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/forte/ontology_specs/metric.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology metric. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import FNdArray
from forte.data.ontology.top import Generics
from typing import Optional

__all__ = [
    "Metric",
    "SingleMetric",
    "NdMetric",
]


@dataclass
class Metric(Generics):
    """
    A base metric entity, all metric entities should inherit from it.
    Attributes:
        metric_name (Optional[str]):
    """

    metric_name: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.metric_name: Optional[str] = None


@dataclass
class SingleMetric(Metric):
    """
    A single metric entity, used to present a metric of one float (e.g. accuracy).
    Attributes:
        value (Optional[float]):
    """

    value: Optional[float]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: Optional[float] = None


@dataclass
class NdMetric(Metric):
    """
    A N-dimensional metric entity, used to present a metric of N-d array (e.g. confusion matrix).
    Attributes:
        value (FNdArray):
    """

    value: FNdArray

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: FNdArray = FNdArray(shape=None, dtype='float')
