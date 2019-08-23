"""
The reader that reads plain text data into Datapacks.
"""
import logging
from typing import Iterator, no_type_check, List, Optional, Any

from forte import config
from forte.data.data_pack import DataPack, ReplaceOperationsType
from forte.data.ontology import base_ontology
from forte.data.readers.file_reader import PackReader

logger = logging.getLogger(__name__)

__all__ = [
    "StringReader",
]


class StringReader(PackReader):
    """:class:`StringReader` is designed to read in a list of string variables.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    @no_type_check
    def __init__(self):
        super().__init__()
        self._ontology = base_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    @staticmethod
    def _cache_key_function(collection):
        return str(collection)

    @staticmethod
    def _collect(dataset: List[str]) -> Iterator[str]:
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        for data in dataset:
            yield data

    def parse_pack(self, collection: Any) -> DataPack:
        config.working_component = self.component_name

        pack = DataPack()

        document = self._ontology.Document(0, len(collection))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(collection)

        config.working_component = None
        return pack
