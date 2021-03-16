# Copyright 2019 The Forte Authors. All Rights Reserved.
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
The reader for reading sentences from text files into MultiPack
"""
import os
from typing import Any, Iterator, Dict, Tuple

from forte.data.data_utils_io import dataset_path_iterator_with_base
from forte.data.multi_pack import MultiPack
from forte.data.base_reader import MultiPackReader
from ft.onto.base_ontology import Sentence

__all__ = [
    "MultiPackSentenceReader"
]


class MultiPackSentenceReader(MultiPackReader):
    r""":class:`MultiPackSentenceReader` is designed to read a directory of
    files and convert each file's contents into a data pack. This class yields a
    multipack with pack ``input_pack_name`` containing the file's contents.
    It additionally packs an empty pack with name ``output_pack_name`` into the
    multipack.
    """

    def _collect(self, text_directory: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator_with_base(text_directory, '')

    def _cache_key_function(self, txt_path: str) -> str:
        return os.path.basename(txt_path)

    def _parse_pack(self,
                    base_and_path: Tuple[str, str]) -> Iterator[MultiPack]:
        base_dir, file_path = base_and_path

        m_pack: MultiPack = MultiPack()

        input_pack_name = self.configs.input_pack_name
        output_pack_name = self.configs.output_pack_name

        text = ""
        offset = 0
        with open(file_path, "r", encoding="utf8") as doc:
            # Remove long path from the beginning.
            doc_id = file_path[
                     file_path.startswith(base_dir) and len(base_dir):]
            doc_id = doc_id.strip(os.path.sep)

            input_pack = m_pack.add_pack(input_pack_name)
            input_pack.pack_name = doc_id

            for line in doc:
                line = line.strip()

                if len(line) == 0:
                    continue

                # add sentence
                Sentence(input_pack, offset, offset + len(line))
                text += line + '\n'
                offset = offset + len(line) + 1

            input_pack.set_text(
                text, replace_func=self.text_replace_operation)
            # Create a output pack without text.
            m_pack.add_pack(output_pack_name)
            yield m_pack

    @classmethod
    def default_configs(cls) -> Dict[str, str]:
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "multipack_sentence_reader"
                "input_pack_name": "input_src",
                "output_pack_name": "output_tgt"
            }

        Here:

        `"name"`: str
            Name of the reader

        `"input_pack_name"`: str
            Name of the input pack. This name can be used to retrieve the input
            pack from the multipack.

        `"output_pack_name"`: str
            Name of the output pack. This name can be used to retrieve the
            output pack from the multipack.
        """
        return {
            "name": "multipack_sentence_reader",
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }
