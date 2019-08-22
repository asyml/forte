"""
The reader that reads plain text data into Datapacks.
"""

from typing import Iterator, Optional, Any
from forte.data.io_utils import dataset_path_iterator
from forte.data.data_pack import DataPack, ReplaceOperationsType
from forte.data.ontology import base_ontology
from forte.data.readers.file_reader import MonoFileReader

__all__ = [
    "PlainTextReader",
]


class PlainTextReader(MonoFileReader):
    """:class:`PlainTextReader` is designed to read in plain text dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self._ontology = base_ontology
        self.define_output_info()

    @staticmethod
    def _cache_key_function(collection):
        return str(collection)

    @staticmethod
    def collect(dir_path: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator(dir_path, ".txt")

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    def parse_pack(self, file_path: str,
                   replace_operations: Optional[ReplaceOperationsType]
                   ) -> DataPack:

        pack = DataPack()
        with open(file_path, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()

        document = self._ontology.Document(0, len(text))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(text, replace_operations)
        pack.meta.doc_id = file_path
        return pack
