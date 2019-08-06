"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Type, List, Union, Generic

import jsonpickle

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.base_pack import PackType
from nlp.pipeline.data.ontology import Entry, base_ontology
from nlp.pipeline.utils import get_full_module_name

__all__ = [
    "BaseReader",
    "PackReader"
]


class BaseReader(Generic[PackType]):
    """The basic data reader class.
    To be inherited by all data readers.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            :meth:`iter()` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            :meth:`iter()` returns a list.
    """
    def __init__(self, lazy: bool = True) -> None:
        self.lazy = lazy
        self._cache_directory: Optional[Path] = None
        self._ontology = base_ontology
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.component_name = get_full_module_name(self)

    @property
    def ontology(self):
        return self._ontology

    def set_ontology(self, ontology):
        self._ontology = ontology
        self.define_output_info()

    @abstractmethod
    def define_output_info(self):
        """
        Define :attr:`output_info` according to the entries and fields the
        reader will generate.
        """
        raise NotImplementedError

    @staticmethod
    def serialize_instance(instance: PackType) -> str:
        """
        Serializes a pack to a string.
        """
        return jsonpickle.encode(instance, unpicklable=True)

    @staticmethod
    def deserialize_instance(string: str) -> PackType:
        """
        Deserializes a pack from a string.
        """
        return jsonpickle.decode(string)

    @abstractmethod
    def iter(self, dataset) -> Union[List[PackType], Iterator[PackType]]:
        """
        An iterator over the entire dataset, yielding all documents processed.
        A suggested design is to call :meth:`read` in a loop to read each
        document.

        Returns:
            If :attr:`lazy` is `True`, returns an iterator of :class:`BasePack`
            objects. Otherwise, returns a list of :class:`BasePack` objects.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, data) -> PackType:
        """
        Read a single document and add entries into the :class:`BasePack`.
        Should **update config.working_component** at the beginning and the end
        of this method.

        Returns:
             one :class:`BasePack` object.
        """
        raise NotImplementedError

    @abstractmethod
    def _instances_from_cache_file(self,
                                   cache_filename: Path):
        raise NotImplementedError

    def cache_data(self, cache_directory: str) -> None:
        """Specify the path to the cache directory.

        After you call this method, the dataset reader will use this
        :attr:`cache_directory` to store a cache of :class:`BasePack` read
        from every document passed to :func:`read`, serialized as one
        string-formatted :class:`BasePack`. If the cache file for a given
        ``file_path`` exists, we read the :class:`BasePack` from the cache
        (using :func:`deserialize_instance`).  If the cache file does not
        exist, we will `create` it on our first pass through the data (using
        :func:`serialize_instance`).
        """
        self._cache_directory = Path(cache_directory)
        Path.mkdir(self._cache_directory, exist_ok=True)

    def _get_cache_location_for_file_path(self,
                                          file_path: str) -> Optional[Path]:
        if self._cache_directory is None:
            return None
        while file_path[-1] == '/':
            file_path = file_path[:-1]
        return Path(f"{self._cache_directory / file_path.split('/')[-1]}.cache")

    def _record_fields(self, pack: PackType):
        """
        Record the fields and entries that this processor add to packs.
        """
        for entry_type, info in self.output_info.items():
            component = self.component_name
            fields: List[str] = []
            if isinstance(info, list):
                fields = info
            elif isinstance(info, dict):
                fields = info["fields"]
                if "component" in info.keys():
                    component = info["component"]
            pack.record_fields(fields, entry_type, component)


class PackReader(BaseReader[DataPack]):
    """
    The basic :class:`DataPack` reader class.
    To be inherited by all :class:`DataPack` readers.
    """

    def _instances_from_cache_file(self,
                                   cache_filename: Path) -> Iterator[DataPack]:
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                pack = self.deserialize_instance(line.strip())
                if not isinstance(pack, DataPack):
                    raise TypeError(f"Pack deserialized from {cache_filename} "
                                    f"is {type(pack)}, but expect {DataPack}")
                yield pack
