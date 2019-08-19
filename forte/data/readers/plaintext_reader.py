"""
The reader that reads plain text data into Datapacks.
"""
import codecs
from typing import Iterator, Any
from forte.data.io_utils import dataset_path_iterator
from forte.data.data_pack import DataPack
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
    def collect(dir_path: str) -> Iterator[Any]:
        return dataset_path_iterator(dir_path, ".txt")

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    def parse_pack(self, file_path: str) -> DataPack:
        pack = DataPack()
        doc = codecs.open(file_path, "rb", encoding="utf8", errors='ignore')
        text = doc.read()

        document = self._ontology.Document(0, len(text))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(text)
        pack.meta.doc_id = file_path
        doc.close()
        return pack
