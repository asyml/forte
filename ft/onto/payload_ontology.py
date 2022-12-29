# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/forte/ontology_specs/payload_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology payload_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import AudioPayload
from forte.data.ontology.top import ImagePayload
from typing import Optional

__all__ = [
    "JpegPayload",
    "SoundFilePayload",
]


@dataclass
class JpegPayload(ImagePayload):
    """
    Attributes:
        extensions (Optional[str]):
        mime (Optional[str]):
        type_code (Optional[str]):
        version (Optional[int]):
        source_type (Optional[int]):
    """

    extensions: Optional[str]
    mime: Optional[str]
    type_code: Optional[str]
    version: Optional[int]
    source_type: Optional[int]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.extensions: Optional[str] = None
        self.mime: Optional[str] = None
        self.type_code: Optional[str] = None
        self.version: Optional[int] = None
        self.source_type: Optional[int] = None


@dataclass
class SoundFilePayload(AudioPayload):
    """
    Attributes:
        source_type (Optional[str]):
        sample_rate (Optional[int]):
        channels (Optional[int]):
        dtype (Optional[str]):
        encoding (Optional[int]):
    """

    source_type: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    dtype: Optional[str]
    encoding: Optional[int]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.source_type: Optional[str] = None
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None
        self.dtype: Optional[str] = None
        self.encoding: Optional[int] = None
