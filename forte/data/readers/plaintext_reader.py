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

    def _cache_key_function(self, collection):
        return str(collection)

    def _collect(self, **kwargs) -> Iterator[Any]:
        return dataset_path_iterator(kwargs['data_source'], ".txt")

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    def text_replace_operation(self, text: str):
        return []

    def parse_pack(self, file_path: str) -> DataPack:
        pack = DataPack()

        with open(file_path, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()

        pack.set_text(text, replace_func=self.text_replace_operation)

        document = self._ontology.Document(0, len(pack.text))  # type: ignore
        pack.add_or_get_entry(document)

        pack.meta.doc_id = file_path
        return pack
