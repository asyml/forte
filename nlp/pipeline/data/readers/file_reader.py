"""
File readers.
"""
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Union

from nlp.pipeline import config
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.base_reader import DataPackReader

logger = logging.getLogger(__name__)

__all__ = [
    "MonoFileReader",
]


class MonoFileReader(DataPackReader):
    """
    :class:`DataPack` reader that reads one data pack from each single file.
    To be inherited by all mono file data readers.
    """

    def iter(self, dir_path: str) -> Union[List[DataPack], Iterator[DataPack]]:
        """
        An iterator over the entire dataset, yielding all documents processed.

        Args:
            dir_path (str): The directory path of the dataset. The reader will
                read all the files according to :meth:`dataset_path_iterator`
                under this directory.
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_path} does not exist.")

        cache_file = None
        if self._cache_directory is not None:
            cache_file = self._get_cache_location_for_file_path(dir_path)

        has_cache = cache_file is not None and cache_file.exists()

        if self.lazy:
            return self._lazy_iter(dir_path, cache_file, has_cache)

        if has_cache:
            logger.info("reading from cache file %s", cache_file)
            return list(
                self._instances_from_cache_file(cache_file))  # type: ignore

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

    def _lazy_iter(self, dir_path: str,
                               cache_file: Optional[Path],
                               has_cache: bool):
        if has_cache:
            logger.info("reading from cache file %s", cache_file)
            yield from self._instances_from_cache_file(  # type: ignore
                cache_file)
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
        An iterator over valid file paths in a directory.

        By default, we iterate through all file paths under ``dir_path``.
        Users can override this function to restrict the returned file paths.
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                yield os.path.join(root, data_file)

    def read(self,
             file_path: str,
             cache_file: Optional[Path] = None,
             read_from_cache: bool = True,
             append_to_cache: bool = False) -> DataPack:
        """
        Read a **single** :class:`DataPack` from original file or from caching
        file. The cache file is supposed to contain only one line corresponding
        to the json format of a :class:`DataPack. If the cache file contains
        multiple lines, only read the :class:`DataPack` in the first line.

        Args:
            file_path (str): The path to the original file to read.
            cache_file (str, optional): The path of the caching file. If
                ``cache_file`` is ``None`` and
                :attr:`self._cache_directory` is not ``None``, use the result
                of :meth:`_get_cache_location_for_file_path`. If both
                ``cache_file`` and :attr:`self._cache_directory`
                are ``None``, will disable cache reading and writing.
            read_from_cache (bool, optional): Decide whether to read from cache
                if cache file exists. By default (`True`), the reader will
                try to read an datapack from the first line of the caching file.
                If `False`, the reader will only read from the original file
                and use the cache file path only for output.
            append_to_cache (bool, optional): Decide whether to append write
                if cache file already exists.  By default (`False`), we
                will overwrite the existing caching file. If `True`, we will
                cache the datapack append to end of the caching file.
        """
        config.working_component = self.component_name
        if cache_file is None and self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)

        if read_from_cache and cache_file and cache_file.exists():
            logger.info("reading from cache file %s", cache_file)
            datapack = next(self._instances_from_cache_file(cache_file))

            if not isinstance(datapack, DataPack):
                raise ValueError(
                    f"No Datapack object read from the given "
                    f"file path {file_path}. "
                )
        else:
            logger.info("reading from original file %s", file_path)
            datapack = self._read_document(file_path)
            self._record_fields(datapack)
            if not isinstance(datapack, DataPack):
                raise ValueError(
                    f"No DataPack object read from the given "
                    f"file path {file_path}. "
                )

            # write to the cache if we need to.
            if cache_file:
                logger.info("Caching datapack to %s", cache_file)
                if append_to_cache:
                    with cache_file.open('a') as cache:
                        cache.write(self.serialize_instance(datapack) + "\n")
                else:
                    with cache_file.open('w') as cache:
                        cache.write(self.serialize_instance(datapack) + "\n")

        config.working_component = None
        return datapack

    @abstractmethod
    def _read_document(self, file_path: str):
        """
        Process the original document. Should be Implemented according to the
        document formant.
        """
        raise NotImplementedError
