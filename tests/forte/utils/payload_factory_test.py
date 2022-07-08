# Copyright 2021 The Forte Authors. All Rights Reserved.
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
from forte.utils.payload_factory import (
    AudioPayloading,
    ImagePayloading,
    PayloadFactory,
)
from ft.onto.payload_ontology import (
    AudioPayload,
    ImagePayload,
    JpegMeta,
    AudioMeta,
)
from forte.data.data_pack import DataPack


class PayloadFactoryTest(unittest.TestCase):
    """
    Test PayloadFactory.
    """

    def setUp(self):

        self.f = PayloadFactory()

    def test_image_payloading(self):
        datapack = DataPack("image")
        img_meta = JpegMeta(datapack)
        img_meta.source_type = "local"

        self.f.register(img_meta)

        # 2. each payloading intialized with a factory
        payloading = ImagePayloading()
        # payload loads the factory with registered meta data
        payloading.load_factory(self.f)

        # payloading = UriImagePayloading()
        fn = payloading.route(img_meta)

        # 3. datapack and payload

        uri = "test.png"
        # uri = "https://assets.website-files.com/6241e60ecd4aa2049d61387c/62576e00dd225cf869b24e0f_61f880d055d4f6f2497fb3cc_symphony-EDITOR-p-1080.jpeg"
        ip = ImagePayload(datapack, 0, uri=uri)
        ip.payloading = payloading
        ip.set_meta(img_meta)  # maybe only store a meta name in ip
        ip.load()
        print(ip.cache)

    def test_audio_payloading(self):
        datapack = DataPack("audio")
        audio_meta = AudioMeta(datapack)
        audio_meta.source_type = "local"
        audio_meta.sample_rate = 44100
        audio_meta.channels = 2
        audio_meta.dtype = "float64"
        audio_meta.encoding = "flac"
        self.f.register(audio_meta)

        # 2. each payloading intialized with a factory
        payloading = AudioPayloading()
        # payload loads the factory with registered meta data
        payloading.load_factory(self.f)

        fn = payloading.route(audio_meta)

        # 3. datapack and payload

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
        ap = AudioPayload(datapack, 0, uri=uri)
        ap.payloading = payloading
        ap.set_meta(audio_meta)  # maybe only store a meta name in ap
        ap.load()
        print("Audio payload data:", ap.cache)
