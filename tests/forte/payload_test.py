# Copyright 2022 The Forte Authors. All Rights Reserved.
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
Unit tests for Payload.
"""
import os
import unittest
import numpy as np
from typing import Dict

from numpy import array_equal
from forte.data.ontology.top import Payload, AudioPayload
from forte.data.data_pack import DataPack


class PayloadTest(unittest.TestCase):
    """
    Test Payload related ontologies like audio.
    """

    def setUp(self):
        self.datapack = DataPack("payload test")

    def test_audio_payload(self):
        class SoundfileAudioPayload(AudioPayload):
            def loading_method(self, path):
                try:
                    import soundfile  # pylint: disable=import-outside-toplevel
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "AudioReader requires 'soundfile' package to be installed."
                        " You can refer to [extra modules to install]('pip install"
                        " forte['audio_ext']) or 'pip install forte"
                        ". Note that additional steps might apply to Linux"
                        " users (refer to "
                        "https://pysoundfile.readthedocs.io/en/latest/#installation)."
                    ) from e
                audio, sample_rate = soundfile.read(file=path)
                audio_data_meta = {"sample_rate": sample_rate}
                return audio, audio_data_meta

        self.datapack.add_entry(
            SoundfileAudioPayload(self.datapack, payload_idx=0)
        )
