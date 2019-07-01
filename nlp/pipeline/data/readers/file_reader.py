"""
File readers.
"""
import logging
import os
from abc import abstractmethod
from typing import Iterator, List
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.base_reader import BaseReader
from nlp.pipeline.data.base_ontology import BaseOntology
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MonoFileReader(BaseReader):
    """Data reader that reads one data pack from each single text files.
    To be inherited by all mono file data readers.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """

    def __init__(self, lazy: bool = True) -> None:
        super().__init__(lazy)

    def dataset_iterator(self, dir_path: str):
        """
        An iterator over the entire dataset, yielding all documents processed.

        Args:
            dir_path (str): The directory path of the dataset. The reader will
                read all the files under this directory.
        """

        if self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(dir_path)
        else:
            cache_file = None

        has_cache = cache_file and os.path.exists(cache_file)

        if self.lazy:
            return self._lazy_dataset_iterator(dir_path, cache_file, has_cache)

        if has_cache:
            logger.info("reading from cache file %s", cache_file)
            return list(self._instances_from_cache_file(cache_file))

        logger.info("reading from original files in %s", dir_path)
        datapacks: List[DataPack] = []
        for file_path in self.dataset_path_iterator(dir_path):
            datapacks.append(
                self.read(
                    file_path,
                    cache_file=cache_file,
                    read_from_cache=False,
                    append_to_cache=True,
                )
            )
        return datapacks

    def _lazy_dataset_iterator(self, dir_path: str,
                               cache_file: str,
                               has_cache: bool):
        if has_cache:
            logger.info("reading from cache file %s", cache_file)
            yield from self._instances_from_cache_file(cache_file)
        else:
            logger.info("reading from original files in %s", dir_path)
            for file_path in self.dataset_path_iterator(dir_path):
                yield self.read(
                    file_path,
                    cache_file=cache_file,
                    read_from_cache=False,
                    append_to_cache=True,
                )

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file paths in a directory
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                yield os.path.join(root, data_file)

    def read(self,
             file_path: str,
             cache_file: str = None,
             read_from_cache: bool = True,
             append_to_cache: bool = False) -> DataPack:
        """
        Read a **single** document from original file or from caching file.
        If the cache file contains multiple lines, only read the datapack
        in the first line. The reader automatically build the link index,
        group index, and sentence-to-entry coverage index.

        Args:
            file_path (str): The path to the original file to read.
            cache_file (str, optional): The path of the caching file. If
                :attr:`cache_file_path` is ``None`` and
                :attr:`self._cache_directory` is not ``None``, use the result
                of :meth:`_get_cache_location_for_file_path`. If both
                :attr:`cache_file_path` and :attr:`self._cache_directory`
                are ``None``, will not read from or write to a caching file.
            read_from_cache (bool, optional): Decide whether to read from cache
                if cache file exists. By default (``True``), the reader will
                try to read an datapack from the first line of the caching file.
                If ``False``, the reader will only read from the original file
                and use the cache file path only for output.
            append_to_cache (bool, optional): Decide whether to append write
                if cache file already exists.  By default (``False``), we
                will overwrite the existing caching file. If ``True``, we will
                cache the datapack append to end of the caching file.
        """
        if cache_file is None and self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)

        if read_from_cache and cache_file and os.path.exists(cache_file):
            logger.info("reading from cache file %s", cache_file)
            datapack = next(self._instances_from_cache_file(cache_file))

            if not isinstance(datapack, DataPack):
                raise ValueError(
                    f"No Datapack object read from the given "
                    f"file path {file_path}. "
                )
        else:
            logger.info("reading from original file %s", file_path)
            self.current_datapack = DataPack()
            self._record_fields()
            datapack = self._read_document(file_path)
            datapack.index.build_coverage_index(
                datapack.annotations, datapack.links, datapack.groups,
                outer_type=BaseOntology.Sentence
            )
            if not isinstance(datapack, DataPack):
                raise ValueError(
                    f"No DataPack object read from the given "
                    f"file path {file_path}. "
                )

            # write to the cache if we need to.
            if cache_file:
                logger.info("Caching datapack to %s", cache_file)
                if append_to_cache:
                    with open(cache_file, "a") as cache:
                        cache.write(self.serialize_instance(datapack) + "\n")
                else:
                    with open(cache_file, "w") as cache:
                        cache.write(self.serialize_instance(datapack) + "\n")

        return datapack

    @abstractmethod
    def _read_document(self, file_path: str):
        """
        Process the original document. Should be Implemented according to the
        document formant.
        """
        raise NotImplementedError

    @abstractmethod
    def _record_fields(self):
        raise NotImplementedError
