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

from forte.data.data_pack import DataPack
from forte.data.ontology.top import load_func
from ft.onto.payload_ontology import (
    JpegPayload,
    SoundFilePayload
)


class OnlineJpegPayload(JpegPayload):
    pass


class LocalJpegPayload(JpegPayload):
    pass


class LocalSoundfilePayload(SoundFilePayload):
    pass


@load_func(OnlineJpegPayload)
def load(payload: OnlineJpegPayload):
    """
    A function that parses payload metadata and prepare and returns a loading function.

    This function is not stored in data store but will be used
    for registering in PayloadFactory.

    Returns:
        a function that reads image data from an url.
    """
    try:
        from PIL import Image
        import requests
        import numpy as np
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ImagePayload reading web file requires `PIL` and"
            "`requests` packages to be installed."
        ) from e

    def read_uri(input_uri):
        # customize this function to read data from uri
        uri_obj = requests.get(input_uri, stream=True)
        pil_image = Image.open(uri_obj.raw)
        return np.asarray(pil_image)

    return read_uri(payload.uri)


@load_func(LocalJpegPayload)
def load(payload: LocalJpegPayload):
    """
    A function that parses payload metadata and prepare and returns a loading function.

    This function is not stored in data store but will be used
    for registering in PayloadFactory.

    Returns:
        a function that reads image data from an url.
    """

    def load_fn(input_uri):
        try:
            import matplotlib.pyplot as plt
            return plt.imread(input_uri)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ImagePayload reading local file requires `matplotlib`"
                "package to be installed."
            ) from e

    return load_fn(payload.uri)


@load_func(LocalSoundfilePayload)
def load(payload: LocalSoundfilePayload):
    try:
        import soundfile  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "AudioPayload requires 'soundfile' package to be installed."
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

    def read_uri(input_uri):
        if payload.encoding is None:  # data type is ".raw"
            return get_first(
                soundfile.read(
                    file=input_uri,
                    samplerate=payload.sample_rate,
                    channels=payload.channels,
                    dtype=payload.dtype,
                )
            )
        else:  # sound file auto-detect the
            return get_first(soundfile.read(file=input_uri))
    return read_uri(payload.uri)


class PayloadDecoratorTest(unittest.TestCase):
    """
    Test PayloadFactory.
    """

    def test_online_image_payload(self):
        datapack = DataPack("image")
        uri = "https://raw.githubusercontent.com/asyml/forte/assets/ocr_tutorial/ocr.jpg"
        payload = OnlineJpegPayload(datapack)
        payload.uri = uri
        payload.load()

        datapack.add_entry(payload)
        self.assertEqual(payload.cache.shape, (539, 810, 3))

    def test_audio_payload(self):
        datapack = DataPack("audio")
        payload = LocalSoundfilePayload(datapack)
        payload.uri = (
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
        payload.load()

        datapack.add_entry(payload)
        self.assertEqual(payload.cache.shape, (74400, ))
