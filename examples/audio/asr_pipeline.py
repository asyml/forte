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
A forte pipeline for automatic speech recognition
"""
import os
import logging
from torch import argmax
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.common.exception import ProcessFlowException
from forte.data.data_pack import DataPack
from forte.data.readers import AudioReader
from forte.pipeline import Pipeline
from forte.processors.base.pack_processor import PackProcessor

from forte.data.ontology.top import Link
from ft.onto.base_ontology import AudioUtterance, Utterance

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class ASRProcessor(PackProcessor):
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
        required_sample_rate: int = 16000
        if input_pack.sample_rate != required_sample_rate:
            raise ProcessFlowException(
                f"A sample rate of {required_sample_rate} Hz is requied by the"
                " pretrained model."
            )

        # tokenize
        input_values = self._tokenizer(
            input_pack.audio, return_tensors="pt", padding="longest"
        ).input_values  # Batch size 1

        # take argmax and decode
        transcription = self._tokenizer.batch_decode(
            argmax(self._model(input_values).logits, dim=-1)
        )

        input_pack.set_text(text=transcription[0])

        # Create annotations on audio and text utterance
        audio_utter: AudioUtterance = AudioUtterance(
            pack=input_pack, begin=0, end=len(input_pack.audio)
        )
        text_utter: Utterance = Utterance(
            pack=input_pack, begin=0, end=len(input_pack.text)
        )
        audio_utter.speaker = text_utter.speaker = "speaker"
        Link(pack=input_pack, parent=audio_utter, child=text_utter)


if __name__ == "__main__":
    audio_path: str = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            os.pardir,
            "data_samples/audio_reader_test"
        )
    )

    # Define and config the Pipeline
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(AudioReader())
    pipeline.add(ASRProcessor())
    pipeline.initialize()

    # Print out the ASR result of each datapack
    for pack in pipeline.process_dataset(audio_path):
        for asr_link in pack.get(Link):
            audio_utter = asr_link.get_parent()
            text_utter = asr_link.get_child()
            logger.info("%s: %s", text_utter.speaker, text_utter.text)
            logger.info("Size of audio payload: %s", audio_utter.audio.shape)
            logger.info("Sample rate: %d", pack.sample_rate)
