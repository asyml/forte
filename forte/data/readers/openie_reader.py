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
The reader that reads Open-IE extractions data into data pack.
Format:
https://github.com/gabrielStanovsky/oie-benchmark/tree/master/oie_corpus
"""
import logging
import os

from typing import Iterator, Any
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Sentence, PredicateMention, \
    PredicateArgument, PredicateLink, Document

__all__ = [
    "OpenIEReader"
]


class OpenIEReader(PackReader):
    r""":class:`OpenIEReader` is designed to read in the Open IE dataset.
    """

    def _collect(self, *args, **kwargs) -> Iterator[Any]:
        # pylint: disable = unused-argument
        r"""Should be called with param ``oie_directory`` which is a path to a
        folder containing json files.

        Args:
            args: args[0] is the directory to the open ie files.
            kwargs:

        Returns: Iterator over files in the path with oie extensions.
        """
        oie_directory = args[0]
        logging.info("Reading .oie from %s", oie_directory)
        return dataset_path_iterator(oie_directory, "oie")

    def _cache_key_function(self, oie_file: str) -> str:
        return os.path.basename(oie_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = self.new_pack()
        with open(file_path, "r", encoding="utf8") as f:
            doc = f.readlines()

        text = ""
        offset = 0

        for line in doc:
            line = line.strip()
            if line != "":
                oie_component = line.split("\t")
                sentence = oie_component[0]

                # Add sentence.
                Sentence(pack, offset, offset + len(sentence))
                offset += len(sentence) + 1
                text += sentence + " "

                predicate = oie_component[1]

                # Add predicate.
                predicate_mention = PredicateMention(pack,
                                                     offset,
                                                     offset + len(predicate))
                offset += len(predicate) + 1
                text += predicate + " "

                for arg in oie_component[2:]:
                    # Add predicate argument.
                    predicate_arg = PredicateArgument(pack,
                                                      offset,
                                                      offset + len(arg))
                    offset += len(arg) + 1
                    text += arg + " "

                    # Add predicate link.
                    PredicateLink(pack, predicate_mention, predicate_arg)

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, len(text))

        pack.pack_name = file_path

        yield pack

    @classmethod
    def default_configs(cls):
        return {}
