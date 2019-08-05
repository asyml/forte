"""
The reader that reads plain text data into Datapacks.
"""
import logging
from pathlib import Path
from typing import Iterator, no_type_check, List, Optional

from nlp.pipeline import config
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers.base_reader import PackReader

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

    def iter(self, dataset: List[str]) -> Iterator[DataPack]:
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        for data in dataset:
            yield self.read(data)

    def read(self, data: str,
             cache_file: Optional[Path] = None,
             append_to_cache: bool = False) -> DataPack:
        config.working_component = self.component_name

        pack = DataPack()
        self._record_fields(pack)

        document = self._ontology.Document(0, len(data))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(data)

        if cache_file is None and self._cache_directory:
            cache_file = self._cache_directory / "dataset.cache"
        # write to the cache if we need to.
        if cache_file:
            logger.info("Caching datapack to %s", cache_file)
            if append_to_cache:
                with cache_file.open('a') as cache:
                    cache.write(self.serialize_instance(pack)
                                + "\n")
            else:
                with cache_file.open('w') as cache:
                    cache.write(self.serialize_instance(pack)
                                + "\n")
        config.working_component = None
        return pack
