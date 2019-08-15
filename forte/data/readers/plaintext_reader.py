"""
The reader that reads plain text data into Datapacks.
"""
import codecs
from typing import List
from forte.data.data_utils import DataUtils
from forte.data.data_pack import DataPack
from forte.data.ontology import base_ontology
from forte.data.readers.file_reader import MultiFileReader

__all__ = [
    "PlainTextReader",
]


class PlainTextReader(MultiFileReader):
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
        util = DataUtils(".txt")
        self.dataset_path_iterator = util.dataset_path_iterator

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    def _read_packs_from_file(self, file_path: str) -> List[DataPack]:

        datapacks = []
        file = codecs.open(file_path, "rb", encoding="utf8", errors='ignore')

        docs = file.read().split('\n')

        for doc in docs:
            pack = DataPack()
            document = self._ontology.Document(0, len(doc))  # type: ignore
            pack.add_or_get_entry(document)
            pack.set_text(doc)
            pack.meta.doc_id = file_path
            datapacks.append(pack)

        file.close()
        return datapacks
