"""
The reader that reads plain text data into Datapacks.
"""
import logging
from typing import Iterator, Optional, List, Union
from pathlib import Path

from forte import config
from forte.data.data_pack import DataPack
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

    # pylint: disable=unused-argument
    def __init__(self,
                 lazy: bool = True,
                 from_cache: bool = False,
                 cache_directory: Optional[Path] = None,
                 append_to_cache: bool = False):
        super().__init__()
        self._ontology = base_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection):
        return "cached_string_file"

    # pylint: disable=no-self-use
    def _collect(self,  # type: ignore
                 string_data: Union[List[str], str]) -> Iterator[str]:
        """
        data_strings should be of type `List[str]`
        which is the list of raw text strings to iterate over
        """
        # This allows the user to pass in either one single string or a list of
        # strings.
        data_strings = [string_data] if isinstance(
            string_data, str) else string_data
        for data in data_strings:
            yield data

    def parse_pack(self, data_source: str) -> DataPack:
        """
        Takes a raw string and converts into a DataPack
        :param data_source: str that contains text of a document
        :return: DataPack containing Document
        """
        config.working_component = self.component_name

        pack = DataPack()

        document = self._ontology.Document(0, len(data_source))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(data_source, replace_func=self.text_replace_operation)

        config.working_component = None
        return pack
