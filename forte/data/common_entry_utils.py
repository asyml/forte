# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Create a few common functions that are frequently used to interact with
built-in ontology data types.
"""
from typing import Optional

from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Utterance

__all__ = [
    'create_utterance',
    'get_last_utterance',
]


def get_last_utterance(
        input_pack: DataPack, target_speaker: str) -> Optional[Utterance]:
    """
    Get the last utterance from a particular speaker. An utterance is an entry
    of type :class:`~ft.onto.base_ontology.Utterance`

    Args:
        input_pack: The data pack to find utterances.
        target_speaker: The name of the target speaker.

    Returns:
        The last Utterance from the speaker if found, None otherwise.
    """
    utterance: Optional[Utterance] = None
    u: Utterance
    for u in input_pack.get(Utterance):
        if u.speaker == target_speaker:
            utterance = u

    return utterance


def create_utterance(input_pack: DataPack, text: str, speaker: str):
    """
    Create an utterance in the datapack. This is composed of two steps:
     1. Append the utterance text to the data pack.
     2. Create :class:`~ft.onto.base_ontology.Utterance` entry on the text.
     3. Set the speaker of the utterance to the provided `speaker`.

    Args:
        input_pack: The data pack to add utterance into.
        text: The text of the utterance.
        speaker: The speaker name to be associated with the utterance.

    """
    input_pack.set_text(input_pack.text + '\n' + text)

    u = Utterance(input_pack,
                  len(input_pack.text) - len(text),
                  len(input_pack.text))
    u.speaker = speaker
