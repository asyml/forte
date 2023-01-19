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

from requests import RequestException

from forte.data.data_pack import DataPack
from forte.data.ontology.top import load_func, AudioPayload
from ft.onto.payload_ontology import JpegPayload, SoundFilePayload


class PillowJpegPayload(JpegPayload):
    pass


@load_func(PillowJpegPayload)
def load(payload: PillowJpegPayload):
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
        try:
            # customize this function to read data from uri
            open_path = requests.get(input_uri, stream=True).raw
        except RequestException:
            open_path = input_uri

        pil_image = Image.open(open_path)
        return np.asarray(pil_image)

    return read_uri(payload.uri)


@load_func(JpegPayload)
def load(payload: JpegPayload):
    def read_uri(input_uri):
        return f"unimplemented parent JpegPayload with {input_uri}"

    return read_uri(payload.uri)


@load_func(AudioPayload)
def load(payload: AudioPayload):
    def read_uri(input_uri):
        return f"unimplemented parent AudioPayload with {input_uri}"

    return read_uri(payload.uri)


@load_func(SoundFilePayload)
def load(payload: SoundFilePayload):
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
        payload = PillowJpegPayload(datapack)
        payload.uri = uri
        payload.load()

        datapack.add_entry(payload)

        self.assertEqual(payload.cache.shape, (539, 810, 3))

    def test_local_image_payload(self):
        datapack = DataPack("image")
        # The difference from the `test_online_image_payload` is that we download the file locally
        uri = "https://raw.githubusercontent.com/asyml/forte/assets/ocr_tutorial/ocr.jpg"
        local_path = "ocr.jpg"
        import urllib.request

        urllib.request.urlretrieve(uri, local_path)

        payload = PillowJpegPayload(datapack)
        payload.uri = local_path
        payload.load()

        datapack.add_entry(payload)

        self.assertEqual(payload.cache.shape, (539, 810, 3))

    def test_load_from_parent(self):
        """
        In this test we try to call the load function of the parent classes.

        For example, we registered PillowJpegPayload, its parent is JpegPayload. The
        behavior is that we will invoke the function registering at the right level.
        """
        datapack = DataPack("load_from_parent")

        # Add a `JpegPayload`, which is the parent of `PillowJpegPayload`
        jpeg_uri = "https://raw.githubusercontent.com/asyml/forte/assets/ocr_tutorial/ocr.jpg"
        jpeg_payload = JpegPayload(datapack)
        jpeg_payload.uri = jpeg_uri
        jpeg_payload.load()
        datapack.add_entry(jpeg_payload)
        self.assertEqual(
            jpeg_payload.cache,
            f"unimplemented parent JpegPayload with {jpeg_uri}",
        )

        # Add a `AudioPayload`, which is the parent of `SoundFilePayload`
        audio_uri = "random_path/doesnt/matter"
        audio_payload = AudioPayload(datapack)
        audio_payload.uri = audio_uri
        audio_payload.load()
        datapack.add_entry(audio_payload)
        self.assertEqual(
            audio_payload.cache,
            f"unimplemented parent AudioPayload with {audio_uri}",
        )

    def test_audio_payload(self):
        datapack = DataPack("audio")
        payload = SoundFilePayload(datapack)
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
        self.assertEqual(payload.cache.shape, (74400,))
