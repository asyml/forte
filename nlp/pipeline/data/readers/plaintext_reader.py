"""
The reader that reads plain text data into Datapacks.
"""
import codecs
import os
from typing import Iterator, no_type_check

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers.file_reader import MonoFileReader

__all__ = [
    "PlainTextReader",
]


class PlainTextReader(MonoFileReader):
    """:class:`PlainTextReader` is designed to read in plain text dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """

    @no_type_check
    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self._ontology = base_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: ["span"],
        }

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing
        .txt files.
        """
        for root, _, files in os.walk(dir_path):
            files.sort()
            for data_file in sorted(files):
                if data_file.endswith(".txt"):
                    yield os.path.join(root, data_file)

    def _read_document(self, file_path: str) -> DataPack:
        if self.current_datapack is None:
            raise ValueError("You shouldn never call _read_document() "
                             "directly. Instead, call read() to read a file "
                             "or dataset_iterator() to read a directory.")
        assert isinstance(self.current_datapack, DataPack)
        doc = codecs.open(file_path, "rb", encoding="utf8", errors='ignore')
        text = doc.read()

        document = self._ontology.Document(0, len(text))  # type: ignore
        self.current_datapack.add_or_get_entry(document)

        self.current_datapack.set_text(text)
        self.current_datapack.meta.doc_id = file_path
        doc.close()
        return self.current_datapack
