"""
Base reader type to be inherited by all readers.
"""
import logging
import os
import pathlib
import jsonpickle
from typing import Iterator, List
from nlp.pipeline.io.interchange import Interchange
from nlp.pipeline.io.base_ontology import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BaseReader:
    """The base reader class to be inherited by all data sources.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """

    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy
        self._cache_directory = None
        self.component_name = f"{__name__}.{self.__class__.__name__}"
        self.current_interchange = None

    def cache_data(self, cache_directory: str) -> None:
        """Specify the path to the cache directory.

        After you call this method, the dataset reader will use this
        :attr:`cache_directory` to store a cache of already-processed
        ``Interchange`` in every document passed to :func:`read`,
        serialized as one string-formatted ``Interchange``. If the cache
        file for a given ``file_path`` exists, we read the ``Interchanges``
        from the cache instead of re-processing the data (using
        :func:`deserialize_instance`).  If the cache file does not
        exist, we will `create` it on our first pass through the data
        (using :func:`serialize_instance`).

        """
        self._cache_directory = pathlib.Path(cache_directory)
        os.makedirs(self._cache_directory, exist_ok=True)

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
            logger.info(f"reading from cache file {cache_file}")
            return list(self._instances_from_cache_file(cache_file))
        else:
            logger.info(f"reading from original files in {dir_path}")
            interchanges: List[Interchange] = []
            for file_path in self.dataset_path_iterator(dir_path):
                interchanges.append(
                    self.read(
                        file_path,
                        cache_file=cache_file,
                        read_from_cache=False,
                        append_to_cache=True,
                    )
                )
            return interchanges

    def _lazy_dataset_iterator(self, dir_path: str, cache_file, has_cache):
        if has_cache:
            logger.info(f"reading from cache file {cache_file}")
            yield from self._instances_from_cache_file(cache_file)
        else:
            logger.info(f"reading from original files in {dir_path}")
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

    def read(
        self,
        file_path: str,
        cache_file: str = None,
        read_from_cache: bool = True,
        append_to_cache: bool = False,
    ) -> Interchange:
        """
        Read a **single** document from original file or from caching file.
        If the cache file contains multiple lines, only read the interchange
        in the first line.

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
                try to read an interchange from the caching file. If ``False``,
                the reader will only read from the original file and use the
                cache file path only for output.
            append_to_cache (bool, optional): Decide whether to append write
                if cache file already exists.  By default (``False``), we
                will overwrite the existing caching file. If ``True``, we will
                cache the interchange append to end of the caching file.
        """
        if cache_file is None and self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)

        if read_from_cache and cache_file and os.path.exists(cache_file):
            logger.info(f"reading from cache file {cache_file}")
            interchange = next(self._instances_from_cache_file(cache_file))

            if not isinstance(interchange, Interchange):
                raise ValueError(
                    f"No Interchange object read from the given "
                    f"file path {file_path}. "
                )
        else:
            logger.info(f"reading from original file {file_path}")
            self.current_interchange = Interchange()
            self._record_fields()
            interchange = self._read_document(file_path)

            if not isinstance(interchange, Interchange):
                raise ValueError(
                    f"No Interchange object read from the given "
                    f"file path {file_path}. "
                )

            # write to the cache if we need to.
            if cache_file:
                logger.info(f"Caching interchange to {cache_file}")
                if append_to_cache:
                    with open(cache_file, "a") as cache:
                        cache.write(self.serialize_instance(interchange) + "\n")
                else:
                    with open(cache_file, "w") as cache:
                        cache.write(self.serialize_instance(interchange) + "\n")

        return interchange

    def _read_document(self, file_path: str):
        """
        Process the original document. Should be Implemented according to the
        document formant.
        """
        raise NotImplementedError

    def _get_cache_location_for_file_path(self, file_path: str) -> str:
        return f"{self._cache_directory / file_path.split('/')[-1]}.cache"

    def _instances_from_cache_file(
        self, cache_filename: str
    ) -> Iterator[Interchange]:
        with open(cache_filename, "r") as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    @staticmethod
    def serialize_instance(instance: Interchange) -> str:
        """
        Serializes an ``Interchange`` to a string.
        """
        return jsonpickle.encode(instance)

    @staticmethod
    def deserialize_instance(string: str) -> Interchange:
        """
        Deserializes an ``Interchange`` from a string.
        """
        return jsonpickle.decode(string)

    def _record_fields(self):
        raise NotImplementedError

    def _add_span_annotation(self, annotation_class, begin, end, **kwargs):
        span = Span(begin, end)
        span_anno = annotation_class(component=self.component_name, span=span)
        for k, v in kwargs.items():
            if not hasattr(span_anno, k):
                raise AttributeError(
                    f"class {annotation_class.__qualname__}"
                    f" has no attribute {k}"
                )
            setattr(span_anno, k, v)

        span_anno = self.current_interchange.add_annotation(span_anno)

        return span_anno

    def _add_group_annotation(self, annotation_class, members, **kwargs):
        group = annotation_class(component=self.component_name)
        for k, v in kwargs.items():
            if not hasattr(group, k):
                raise AttributeError(
                    f"class {annotation_class.__qualname__}"
                    f" has no attribute {k}"
                )
            setattr(group, k, v)
        for m in members:
            group.add_member(m)

        group = self.current_interchange.add_annotation(group)

        return group

    def _add_link_annotation(self, annotation_class, parent, child, **kwargs):
        link = annotation_class(component=self.component_name)
        for k, v in kwargs.items():
            if not hasattr(link, k):
                raise AttributeError(
                    f"class {annotation_class.__qualname__}"
                    f" has no attribute {k}"
                )
            setattr(link, k, v)

        link.set_parent(parent)
        link.set_child(child)

        link = self.current_interchange.add_annotation(link)

        return link
