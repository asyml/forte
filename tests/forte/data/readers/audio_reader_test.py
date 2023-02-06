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
Unit tests for AudioReader.
"""
import os
import unittest
from typing import Dict
from forte.data import Modality
from torch import argmax
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.common.exception import ProcessFlowException
from forte.data.data_pack import DataPack
from forte.data.readers import AudioReader
from forte.pipeline import Pipeline
from forte.processors.base.pack_processor import PackProcessor


class TestASRProcessor(PackProcessor):
    """
    An audio processor for automatic speech recognition.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # Initialize tokenizer and model
        pretrained_model: str = "facebook/wav2vec2-base-960h"
        self._tokenizer = Wav2Vec2Processor.from_pretrained(pretrained_model)
        self._model = Wav2Vec2ForCTC.from_pretrained(pretrained_model)

    def _process(self, input_pack: DataPack):
        ap = input_pack.get_payload_at(Modality.Audio, 0)
        sample_rate = ap.sample_rate
        audio_data = ap.cache
        required_sample_rate: int = 16000
        if sample_rate != required_sample_rate:
            raise ProcessFlowException(
                f"A sample rate of {required_sample_rate} Hz is requied by the"
                " pretrained model."
            )

        # tokenize
        input_values = self._tokenizer(
            audio_data, return_tensors="pt", padding="longest"
        ).input_values  # Batch size 1

        # take argmax and decode
        transcription = self._tokenizer.batch_decode(
            argmax(self._model(input_values).logits, dim=-1)
        )

        input_pack.set_text(text=transcription[0])


class AudioReaderPipelineTest(unittest.TestCase):
    """
    Test AudioReader by running audio processing pipelines
    """

    def setUp(self):
        self._test_audio_path: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
                "data_samples/audio_reader_test",
            )
        )
        # Define and config the Pipeline
        self._pipeline = Pipeline[DataPack]()
        self._pipeline.set_reader(AudioReader())
        self._pipeline.add(TestASRProcessor())
        self._pipeline.initialize()

    def test_asr_pipeline(self):
        target_transcription: Dict[str, str] = {
            self._test_audio_path
            + "/test_audio_0.flac": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
            self._test_audio_path
            + "/test_audio_1.flac": (
                "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING "
                "THAN HIS MATTER"
            ),
        }

        # Verify the ASR result of each datapack
        for pack in self._pipeline.process_dataset(self._test_audio_path):
            self.assertEqual(pack.text, target_transcription[pack.pack_name])


if __name__ == "__main__":
    unittest.main()
