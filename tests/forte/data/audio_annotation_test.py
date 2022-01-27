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
import soundfile
from typing import Dict
from numpy import array_equal

from forte.pipeline import Pipeline
from forte.common.exception import ProcessExecutionException
from forte.processors.base.pack_processor import PackProcessor
from forte.data.data_pack import DataPack
from forte.data.readers import AudioReader
from forte.data.ontology.top import (
    Annotation, AudioAnnotation, Generics, Group, Link
)
from ft.onto.base_ontology import Recording, AudioUtterance, Utterance


class RecordingProcessor(PackProcessor):
    """
    A processor to add a Recording ontology to the whole audio data.
    """
    def _process(self, input_pack: DataPack):
        Recording(
            pack=input_pack, begin=0, end=len(input_pack.audio)
        )


class DummyGroup(Group):
    """
    A dummpy `Group` ontology that sets the type of members to
    `AudioAnnotation`.
    """

    MemberType = AudioAnnotation


class DummyLink(Link):
    """
    A dummpy `Link` ontology that sets the type of parent and child to
    `AudioAnnotation`.
    """

    ParentType = AudioAnnotation
    ChildType = AudioAnnotation


class TextUtteranceProcessor(PackProcessor):
    """
    A processor that sets a random text and adds an `Utterance`
    annotation to input datapack.
    """

    def _process(self, input_pack: DataPack):
        input_pack.set_text("test text")
        Utterance(pack=input_pack, begin=0, end=len(input_pack.text))


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

        # Create a series of annotations for test case
        recording: Recording = input_pack.get_single(Recording)
        Group(pack=input_pack, members=(audio_utter, recording))
        Link(pack=input_pack, parent=recording, child=audio_utter)
        DummyGroup(pack=input_pack, members=(audio_utter, recording))
        DummyLink(pack=input_pack, parent=recording, child=audio_utter)

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
        self._pipeline.add(TextUtteranceProcessor())
        self._pipeline.initialize()

        audio_file_names = os.listdir(self._test_audio_path)

        self.audio_data1, _ = soundfile.read(os.path.join(self._test_audio_path, "test_audio_1.flac"), always_2d = True)
        self.audio_data2, _ = soundfile.read(os.path.join(self._test_audio_path, "test_audio_0.flac"), always_2d = True)

    def test_audio_annotation(self):

        # Test `DataPack.get_span_audio()` with None audio payload
        with self.assertRaises(ProcessExecutionException):
            pack: DataPack = DataPack()
            pack.set_text("test text")
            pack.get_span_audio(begin=0, end=1)
        idx = 0
        # Verify the annotations of each datapack
        for pack in self._pipeline.process_dataset(self._test_audio_path):
            # test get all audio annotation
            # test get selective fields data from subclass of AudioAnnotation
            data = list(pack.get_data(AudioAnnotation))
            audio_data = [d["context"] for d in data]
            for datum in audio_data:
                # import pdb; pdb.set_trace()
                # print('ddd')
                if idx == 0:
                    self.assertTrue(array_equal(self.audio_data1, datum))
                elif idx == 1:
                    self.assertTrue(array_equal(self.audio_data1[200:35000], datum))
                elif idx == 2:
                    self.assertTrue(array_equal(self.audio_data1[35200:72000], datum))
                elif idx == 3:
                    self.assertTrue(array_equal(self.audio_data2, datum))
                idx += 1

            data = list(pack.get_data(AudioAnnotation,
                                     {Recording:
                                         {"fields": ["recording_class"]},
                                    AudioUtterance:
                                        {"fields": ["speaker"]}}
                                    ))
            # check get_data requests parameter effect
            # by checking if requested fields exist when the datum is for
            # the correct Entry.
            for datum in data:
                if 'Recording' in datum.keys():
                    import pdb; pdb.set_trace()
                    print('')
                    
                    self.assertTrue("recording_class" in datum['Recording'].keys())
                if 'AudioUtterance' in datum.keys():
                    self.assertTrue("speaker" in datum['AudioUtterance'].keys())

            # Check Recording
            recordings = list(pack.get(Recording))
            self.assertEqual(len(recordings), 1)
            self.assertTrue(array_equal(recordings[0].audio, pack.audio))

            # Check serialization/deserialization of AudioAnnotation
            new_pack = DataPack.from_string(pack.to_string())
            self.assertEqual(new_pack.audio_annotations, pack.audio_annotations)

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

            # Check `AudioAnnotation.get(Group/Link/Generics)`. Note that only
            # `DummyGroup` and `DummyLink` entries can be retrieved because
            # they have the correct type of MemberType/ParentType/ChildType.
            for entry_type in (Group, Link):
                self.assertEqual(
                    len(list(recordings[0].get(entry_type))),
                    len(self._test_configs)
                )
            self.assertEqual(len(list(recordings[0].get(Generics))), 0)

            # Check operations with mixing types of entries.
            self.assertEqual(len(list(pack.get(Utterance))), 1)
            utter: Utterance = pack.get_single(Utterance)
            self.assertEqual(len(list(utter.get(AudioAnnotation))), 0)
            self.assertEqual(len(list(recordings[0].get(Utterance))), 0)

            # Verify the new conditional branches in DataPack.get() when dealing
            # with empty annotation/audio_annotation list.
            empty_pack: DataPack = DataPack()
            self.assertEqual(len(list(empty_pack.get(
                entry_type=Annotation, range_annotation=utter
            ))), 0)
            self.assertEqual(len(list(empty_pack.get(
                entry_type=AudioAnnotation, range_annotation=recordings[0]
            ))), 0)

            # Check `DataPack.delete_entry(AudioAnnotation)`
            for audio_annotation in list(pack.get(AudioAnnotation)):
                pack.delete_entry(audio_annotation)
            self.assertEqual(len(list(pack.all_audio_annotations)), 0)


if __name__ == "__main__":
    unittest.main()
