# ***automatically_generated***
# ***source json:forte/ontology_specs/payload_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology payload_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack

from forte.data.ontology.top import Payload, TextPayload, AudioPayload, ImagePayload
from typing import Optional

__all__ = [
    "JpegMeta",
    "AudioMeta",
    "JpegPayload",
]



# @dataclass
# class JpegMeta(Meta):
#     """
#     Attributes:
#         extension (Optional[str]):
#         mime (Optional[str]):
#         type_code (Optional[str]):
#         version (Optional[str]):
#         source_type (Optional[str]):
#     """

#     extension: Optional[str]
#     mime: Optional[str]
#     type_code: Optional[str]
#     version: Optional[str]
#     source_type: Optional[str]

#     def __init__(self, pack: DataPack):
#         super().__init__(pack)
#         self.extension: Optional[str] = None
#         self.mime: Optional[str] = None
#         self.type_code: Optional[str] = None
#         self.version: Optional[str] = None
#         self.source_type: Optional[str] = None


# @dataclass
# class AudioMeta(Meta):
#     """
#     Attributes:
#         sample_rate (Optional[int]):
#         channels (Optional[int]):
#         bits_per_sample (Optional[int]):
#         duration (Optional[float]):
#         bitrate (Optional[int]):
#         encoding (Optional[str]):
#         source_type (Optional[str]):
#         dtype (Optional[str]):
#     """

#     sample_rate: Optional[int]
#     channels: Optional[int]
#     bits_per_sample: Optional[int]
#     duration: Optional[float]
#     bitrate: Optional[int]
#     encoding: Optional[str]
#     source_type: Optional[str]
#     dtype: Optional[str]

#     def __init__(self, pack: DataPack):
#         super().__init__(pack)
#         self.sample_rate: Optional[int] = None
#         self.channels: Optional[int] = None
#         self.bits_per_sample: Optional[int] = None
#         self.duration: Optional[float] = None
#         self.bitrate: Optional[int] = None
#         self.encoding: Optional[str] = None
#         self.source_type: Optional[str] = None
#         self.dtype: Optional[str] = None


@dataclass
class JpegPayload(ImagePayload):
    """
    Attributes:
        meta (Optional[JpegMeta]):
    """


    def __init__(
        self, pack: DataPack, payload_idx: int = 0, uri: Optional[str] = None
    ):
        super().__init__(pack, payload_idx, uri)
        self.extension: Optional[str] = None
        self.mime: Optional[str] = None
        self.type_code: Optional[str] = None
        self.version: Optional[str] = None
        self.source_type: Optional[str] = None