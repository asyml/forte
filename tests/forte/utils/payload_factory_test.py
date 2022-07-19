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
Unit test for payload factory.
"""
import os
import unittest
from ft.onto.payload_ontology import (
    JpegPayload,
    SoundFilePayload
)
from forte.data.data_pack import DataPack
from forte.utils.payload_factory import register

@register
class OnlineJpegPayload(JpegPayload):
    def loading_fn(self):
        """
        A function that parses payload meta data and prepare and returns a loading function.
        
        This function is not stored in data store but will be used
        for registering in PayloadFactory.
        
        Returns:
            a function that reads image data from an url.
        """
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
            import requests  # pylint: disable=import-outside-toplevel
            import numpy as np
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ImagePayloading reading web file requires `PIL` and"
                "`requests` packages to be installed."
            ) from e

        def read_uri(uri):
            # customize this function to read data from uri
            uri_obj = requests.get(uri, stream=True)
            pil_image = Image.open(uri_obj.raw)
            return np.asarray(pil_image)

        return read_uri

@register
class LocalJpegPayload(JpegPayload):
    def loading_fn(self):
        """
        A function that parses payload meta data and prepare and returns a loading function.
        
        This function is not stored in data store but will be used
        for registering in PayloadFactory.
        
        Returns:
            a function that reads image data from an url.
        """
        try:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ImagePayloading reading local file requires `matplotlib`"
                "package to be installed."
            ) from e
        return plt.imread

@register
class LocalSoundfilePayload(SoundFilePayload):
    def loading_fn(self):
        try:
            import soundfile  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "AudioPayloading requires 'soundfile' package to be installed."
                " You can refer to [extra modules to install]('pip install"
                " forte['audio_ext']) or 'pip install forte"
                ". Note that additional steps might apply to Linux"
                " users (refer to "
                "https://pysoundfile.readthedocs.io/en/latest/#installation)."
            ) from e

        def get_first(
            seq,
        ):  # takes the first item as soundfile returns a tuple of (data, samplerate)
            return seq[0]

        def read_uri(uri):
            if self.encoding is None:  # data type is ".raw"
                return get_first(
                    soundfile.read(
                        file=uri,
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype=self.dtype,
                    )
                )
            else:  # sound file auto detect the
                return get_first(soundfile.read(file=uri))

        return read_uri

class PayloadFactoryTest(unittest.TestCase):
    """
    Test PayloadFactory.
    """

    def setUp(self):
        pass

    def test_online_image_payload(self):
        datapack = DataPack("image")
        uri = "https://assets.website-files.com/6241e60ecd4aa2049d61387c/62576e00dd225cf869b24e0f_61f880d055d4f6f2497fb3cc_symphony-EDITOR-p-1080.jpeg"
        payload = OnlineJpegPayload(datapack, uri=uri)        
        payload.loading_fn()(uri)

    def test_audio_payload(self):
        datapack = DataPack("audio")
        uri = "https://assets.website-files.com/6241e60ecd4aa2049d61387c/62576e00dd225cf869b24e0f_61f880d055d4f6f2497fb3cc_symphony-EDITOR-p-1080.jpeg"
        payload = LocalSoundfilePayload(datapack, uri=uri)    
        uri = (
            os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    os.pardir,
                    os.pardir,
                    os.pardir,
                    "data_samples/audio_reader_test",
                )
            )
            + "/test_audio_0.flac"
        )    
        payload.loading_fn()(uri)
