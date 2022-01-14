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
Unit tests for AudioAnnotation.
"""
import os
import unittest
from typing import Dict
from numpy import array_equal

from forte.pipeline import Pipeline
from forte.processors.base.pack_processor import PackProcessor
from forte.data.data_pack import DataPack
from forte.data.readers import AudioReader
from forte.data.ontology.top import AudioAnnotation
from ft.onto.base_ontology import Recording, AudioUtterance


class RecordingProcessor(PackProcessor):
    """
    A processor to add a Recording ontology to the whole audio data.
    """
    def _process(self, input_pack: DataPack):
        Recording(
            pack=input_pack, begin=0, end=len(input_pack.audio)
        )

class AudioUtteranceProcessor(PackProcessor):
    """
    A processor to add an AudioUtterance annotation to the specified span of
    audio payload.
    """
    def _process(self, input_pack: DataPack):
        audio_utter: AudioUtterance = AudioUtterance(
            pack=input_pack,
            begin=self.configs.begin,
            end=self.configs.end
        )
        audio_utter.speaker = self.configs.speaker

    @classmethod
    def default_configs(cls) -> Dict:
        return {"begin": 0, "end": 0, "speaker": "ai"}


class AudioAnnotationTest(unittest.TestCase):
    """
    Test AudioAnnotation related ontologies like Recording and AudioUtterance.
    """

    def setUp(self):
        self._test_audio_path: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                "data_samples/audio_reader_test"
            )
        )

        self._test_configs = {
            "Alice": {"begin": 200, "end": 35000},
            "Bob": {"begin": 35200, "end": 72000}
        }

        # Define and config the Pipeline
        self._pipeline = Pipeline[DataPack]()
        self._pipeline.set_reader(AudioReader(), config={
            "read_kwargs": {"always_2d": "True"}
        })
        self._pipeline.add(RecordingProcessor())
        for speaker, span in self._test_configs.items():
            self._pipeline.add(
                AudioUtteranceProcessor(), config={"speaker": speaker, **span}
            )
        self._pipeline.initialize()

    def test_audio_annotation(self):

        # Verify the annotations of each datapack
        for pack in self._pipeline.process_dataset(self._test_audio_path):
            
            # Check Recording
            recordings = list(pack.get(Recording))
            self.assertEqual(len(recordings), 1)
            self.assertTrue(array_equal(recordings[0].audio, pack.audio))

            # Check total number of AudioAnnotations which should be 3
            # (1 Recording + 2 AudioUtterance).
            self.assertEqual(pack.num_audio_annotations, 3)
            
            # Check `DataPack.get(AudioUtterance)` and
            # `AudioAnnotation.get(AudioUtterance)`
            for object in (pack, recordings[0]):
                audio_utters = list(object.get(AudioUtterance))
                self.assertEqual(len(audio_utters), len(self._test_configs))

                for audio_utter in audio_utters:
                    configs: Dict = self._test_configs[audio_utter.speaker]
                    self.assertTrue(array_equal(
                        audio_utter.audio,
                        pack.audio[configs["begin"]:configs["end"]]
                    ))

            # Check `DataPack.delete_entry(AudioAnnotation)`
            for audio_annotation in list(pack.get(AudioAnnotation)):
                pack.delete_entry(audio_annotation)
            self.assertEqual(len(list(pack.all_audio_annotations)), 0)


if __name__ == "__main__":
    unittest.main()
