"""
The reader that reads plain text data into Datapacks.
"""
import logging
from typing import Iterator, List, Union

from forte import config
from forte.data.data_pack import DataPack
from forte.data.ontology import base_ontology
from forte.data.readers.base_reader import PackReader

logger = logging.getLogger(__name__)

__all__ = [
    "StringReader",
]


class StringReader(PackReader):
    """
    :class:`StringReader` is designed to read in a list of string variables.
    """

    def __init__(self):
        super().__init__()
        self._ontology = base_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection) -> str:
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

        document = self._ontology.Document(  # type: ignore
            pack, 0, len(data_source))
        pack.add_or_get_entry(document)

        pack.set_text(data_source, replace_func=self.text_replace_operation)

        config.working_component = None
        return pack
