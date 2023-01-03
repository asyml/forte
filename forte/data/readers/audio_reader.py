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
The reader that reads audio files into Datapacks.
"""
import os
from typing import Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader

__all__ = [
    "AudioReader",
]


class AudioReader(PackReader):
    r""":class:`AudioReader` is designed to read in audio files."""

    def __init__(self):
        super().__init__()
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
        self.soundfile = soundfile

    def _collect(self, audio_directory) -> Iterator[Any]:
        r"""Should be called with param ``audio_directory`` which is a path to a
        folder containing audio files.

        Args:
            audio_directory: audio directory containing the files.

        Returns: Iterator over paths to audio files
        """
        return dataset_path_iterator(audio_directory, self.configs.file_ext)

    def _cache_key_function(self, audio_file: str) -> str:
        return os.path.basename(audio_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack: DataPack = DataPack()

        # Read in audio data and store in DataPack
        audio, sample_rate = self.soundfile.read(
            file=file_path, **(self.configs.read_kwargs or {})
        )
        pack.set_audio(audio=audio, sample_rate=sample_rate)
        pack.pack_name = file_path

        yield pack

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for audio reader.

        Here:

          - file_ext (str): The file extension to find the target audio files
             under a specific directory path. Default value is ".flac".

          - read_kwargs (dict): A dictionary containing all the keyword
             arguments for `soundfile.read` method. For details, refer to
             https://pysoundfile.readthedocs.io/en/latest/#soundfile.read.
             Default value is None.

        Returns: The default configuration of audio reader.
        """
        return {"file_ext": ".flac", "read_kwargs": None}
