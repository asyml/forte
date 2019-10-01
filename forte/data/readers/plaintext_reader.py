"""
The reader that reads plain text data into Datapacks.
"""
import os
from typing import Iterator, Any

from forte.data.data_pack import DataPack
from forte.data.io_utils import dataset_path_iterator
from forte.data.ontology import base_ontology
from forte.data.readers.base_reader import PackReader

__all__ = [
    "PlainTextReader",
]


class PlainTextReader(PackReader):
    """
    :class:`PlainTextReader` is designed to read in plain text dataset.
    """

    def __init__(self):
        super().__init__()
        self._ontology = base_ontology

    # pylint: disable=no-self-use
    def _collect(self, text_directory) -> Iterator[Any]:  # type: ignore
        """
        Should be called with param `data_source`
        which is a path to a folder containing txt files
        :param text_directory: text directory containing the files.
        :return: Iterator over paths to .txt files
        """
        return dataset_path_iterator(text_directory, ".txt")

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def define_output_info(self):
        return {
            self._ontology.Document: [],
        }

    # pylint: disable=no-self-use,unused-argument
    def text_replace_operation(self, text: str):
        return []

    def parse_pack(self, file_path: str) -> DataPack:
        pack = DataPack()

        with open(file_path, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()

        pack.set_text(text, replace_func=self.text_replace_operation)

        document = self._ontology.Document(pack,  # type: ignore
                                           0, len(pack.text))
        pack.add_or_get_entry(document)

        pack.meta.doc_id = file_path
        return pack
