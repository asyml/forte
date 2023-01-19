# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Writers are simply processors with the side-effect to write to the disk.
This file provide some basic writer implementations.
"""
import logging
import os
import posixpath
from abc import abstractmethod, ABC
from typing import Optional, Any, Dict

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.base_pack import BasePack
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base.pack_processor import (
    PackProcessor,
    MultiPackProcessor,
)
from forte.utils.utils_io import maybe_create_dir, ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    "PackWriter",
    "MultiPackWriter",
]


def write_pack(
    input_pack: BasePack,
    output_dir: str,
    sub_path: str,
    indent: Optional[int] = None,
    zip_pack: bool = False,
    overwrite: bool = False,
    drop_record: bool = False,
    serialize_method: str = "json",
) -> str:
    """
    Write a pack to a path.

    Args:
        input_pack: A Pack to be written.
        output_dir: The output directory.
        sub_path: The file name for this pack.
        indent: Whether to format JSON with an indent.
        zip_pack: Whether to zip the output JSON.
        overwrite: Whether to overwrite the file if already exists.
        drop_record: Whether to drop the creation records in the serialization.
        serialize_method: The method used to serialize the data. Current
          available options are `json`, `jsonpickle` and `pickle`.
          Default is `json`.

    Returns:
        If successfully written, will return the path of the output file.
        otherwise, will return None.

    """
    output_path = os.path.join(output_dir, sub_path)

    if overwrite or not os.path.exists(output_path):
        ensure_dir(output_path)
        input_pack.serialize(
            output_path,
            zip_pack=zip_pack,
            drop_record=drop_record,
            serialize_method=serialize_method,
            indent=indent,
        )
    else:
        logging.info("Will not overwrite existing path %s", output_path)

    logging.info("Writing a pack to %s", output_path)
    return output_path


class PackWriter(PackProcessor, ABC):
    def __init__(self):
        super().__init__()
        self._zip_pack: bool = False
        self._indent: Optional[int] = None
        self._suffix: str = ""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if not configs.output_dir:
            raise NotADirectoryError(
                "Root output directory is not defined "
                "correctly in the configs."
            )

        if not os.path.exists(configs.output_dir):
            os.makedirs(configs.output_dir)

        self._zip_pack = configs.zip_pack
        self._indent = configs.indent

        if self.configs.serialize_method in ("jsonpickle", "json"):
            self._suffix = ".json.gz" if self._zip_pack else ".json"
        else:
            self._suffix = ".pickle.gz" if self._zip_pack else ".pickle"

    @abstractmethod
    def sub_output_path(self, pack: DataPack) -> Optional[str]:
        r"""Allow defining output path using the information of the pack. If
        `None` is returned,

        Args:
            pack: The input datapack.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for writer.

        Here:
          - output_dir (str): the directory for writing the result.

          - zip_pack (bool): whether to zip the data pack. The default value is
             False.

          - indent (int): None not indented, if larger than 0, the JSON
             files will be written in the with the provided indention. The
             default value is None.

          - drop_record: whether to drop the creation records in the data pack,
             the default value is False.

          - serialize_method: The method used to serialize the data. Current
              available options are `json`, `jsonpickle` and `pickle`. Default is
              "json".

        Returns: The default configuration of this writer.
        """
        return {
            "output_dir": None,
            "zip_pack": False,
            "indent": None,
            "drop_record": False,
            "serialize_method": "json",
        }

    def _process(self, input_pack: DataPack):
        sub_path = self.sub_output_path(input_pack)
        if sub_path is not None and not sub_path == "":
            # Sub path could be empty, which we will skip writing the file.
            maybe_create_dir(self.configs.output_dir)
            write_pack(
                input_pack,
                self.configs.output_dir,
                sub_path,
                self.configs.indent,
                self.configs.zip_pack,
                self.configs.overwrite,
                self.configs.drop_record,
                self.configs.serialize_method,
            )


class MultiPackWriter(MultiPackProcessor):
    pack_base_out = "packs"
    multi_base = "multi"
    pack_idx = "pack.idx"
    multi_idx = "multi.idx"

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init,consider-using-with
        super().initialize(resources, configs)

        if self.configs.output_dir is None:
            raise ProcessorConfigError(
                "`output_dir` is not specified for the writer."
            )

        pack_paths = os.path.join(self.configs.output_dir, self.pack_idx)
        ensure_dir(pack_paths)
        self.pack_idx_out = open(pack_paths, "w", encoding="utf-8")

        multi_index = os.path.join(self.configs.output_dir, self.multi_idx)
        ensure_dir(multi_index)
        self.multi_idx_out = open(multi_index, "w", encoding="utf-8")

        if self.configs.serialize_method in ("jsonpickle", "json"):
            self._suffix = ".json.gz" if self.configs.zip_pack else ".json"
        else:
            self._suffix = ".pickle.gz" if self.configs.zip_pack else ".pickle"

    def multipack_name(self, pack: MultiPack) -> str:
        r"""Allow defining output path using the information of the multi-pack.
        Extending this path allows one to specify the output file name. Default
        value is a name including the `pack_id` of this multi-pack.

        Args:
            pack: The input multi-pack.
        """
        return f"multi_pack_{pack.pack_id}"

    def _process(self, input_pack: MultiPack):
        multi_out_dir = os.path.join(self.configs.output_dir, self.multi_base)
        pack_out_dir = os.path.join(self.configs.output_dir, self.pack_base_out)

        for pack in input_pack.packs:
            pack_out = write_pack(
                pack,
                pack_out_dir,
                str(pack.pack_id) + self._suffix,
                self.configs.indent,
                self.configs.zip_pack,
                self.configs.overwrite,
                self.configs.drop_record,
                self.configs.serialize_method,
            )

            self.pack_idx_out.write(
                f"{pack.pack_id}\t"
                f"{posixpath.relpath(pack_out, self.configs.output_dir)}\n"
            )

        multi_out = write_pack(
            input_pack,
            multi_out_dir,
            self.multipack_name(input_pack) + self._suffix,
            self.configs.indent,
            self.configs.zip_pack,
            self.configs.overwrite,
            self.configs.drop_record,
            self.configs.serialize_method,
        )

        self.multi_idx_out.write(
            f"{input_pack.pack_id}\t"
            f"{posixpath.relpath(multi_out, self.configs.output_dir)}\n"
        )

    def finish(self, _):
        self.pack_idx_out.close()
        self.multi_idx_out.close()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "output_dir": None,
            "zip_pack": False,
            "indent": None,
            "drop_record": False,
            "serialize_method": "json",
        }
