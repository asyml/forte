#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
import os
import shutil
from typing import Union, Dict, Iterator, List

from forte.data.readers.base_reader import PackReader
from forte.data.data_pack import DataPack
from forte.data.readers.deserialize_reader import DirPackReader
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.processors.writers import PackNameJsonPackWriter

logger = logging.getLogger(__name__)


class DataPackLoader():
    def __init__(self,
                 reader: PackReader,
                 cache_dir: str,
                 config: Union[Config, Dict]):
        self._config = Config(config, default_hparams=self.default_configs())
        self._validate_config()

        # Update cache dir based on input
        self._config.cache_writer.output_dir = cache_dir

        self._reader: PackReader = reader
        self._data_packs: List[DataPack] = []

        if self._config.cache:
            self._cache_writer = PackNameJsonPackWriter()
            self._cache_writer.initialize(Resources(),
                                      Config(self._config.cache_writer,
                                             default_hparams=None))

        self._cache_reader = DirPackReader()

    @staticmethod
    def default_configs():
        return {
            "cache_writer": {
                "output_dir": ".data_pack_cache/",
                "indent": 2,
                "overwrite": True,
                "zip_pack": False,
                "drop_record": False
            },
            "cache": True,
            "read_from_cache": False,
            "src_dir": None,
            "clear_cache_after_finish": False
        }

    def _validate_config(self):
        # TODO: validate config
        pass

    def _write_data_pack(self, data_pack: DataPack):
        # TODO: currently access private method. Any better way?
        self._cache_writer._process(data_pack)

    def load(self) -> Iterator[DataPack]:
        if not self._data_packs:
            if self._config.read_from_cache:
                logger.info("Read data packs from disk cache")
                # Read data packs from disk cache
                cache_dir = self._config.cache_writer.output_dir
                for data_pack in self._cache_reader.iter(cache_dir):
                    self._data_packs.append(data_pack)
                    yield data_pack
            else:
                logger.info("Parse data packs by reader")
                # Ask reader to parse data packs
                for data_pack in self._reader.iter(self._config.src_dir):
                    if self._config.cache:
                        assert hasattr(self, "_cache_writer"), \
                            "Missing cache writer while caching"
                        logger.info("Cache parses data packs to disk")
                        self._write_data_pack(data_pack)
                    self._data_packs.append(data_pack)
                    yield data_pack
        else:
            logger.info("Read data packs from memory cache")
            # data packs already stored in memory
            for data_pack in self._data_packs:
                yield data_pack

    def finish(self):
        if self._config.clear_cache_after_finish:
            cache_dir = self._config.cache_writer.output_dir
            if os.path.exists(cache_dir):
                logger.info("Remove cache directory: %s", cache_dir)
                shutil.rmtree(cache_dir)