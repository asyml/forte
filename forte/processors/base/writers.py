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
import gzip
import json
import logging
import os
from abc import abstractmethod, ABC
from typing import Optional, Any, Dict

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base.base_processor import BaseProcessor
from forte.utils.utils_io import maybe_create_dir, ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    'JsonPackWriter',
    'MultiPackWriter',
]


def write_pack(input_pack: PackType, output_dir: str, sub_path: str,
               indent: Optional[int] = None, zip_pack: bool = False,
               overwrite: bool = False) -> str:
    """
    Write a pack to a path.

    Args:
        input_pack: A Pack to be written.
        output_dir: The output directory.
        sub_path: The file name for this pack.
        indent: Whether to format JSON with an indent.
        zip_pack: Whether to zip the output JSON.
        overwrite: Whether to overwrite the file if already exists.

    Returns:
        If successfully written, will return the path of the output file.
        otherwise, will return None.

    """
    output_path = os.path.join(output_dir, sub_path) + '.json'
    if overwrite or not os.path.exists(output_path):
        if zip_pack:
            output_path = output_path + '.gz'

        ensure_dir(output_path)

        out_str: str = input_pack.serialize()

        if indent:
            out_str = json.dumps(json.loads(out_str), indent=indent)

        if zip_pack:
            with gzip.open(output_path, 'wt') as out:
                out.write(out_str)
        else:
            with open(output_path, 'w') as out:
                out.write(out_str)

    logging.info("Writing a pack to %s", output_path)
    return output_path


class JsonPackWriter(BaseProcessor[DataPack], ABC):
    def __init__(self):
        super().__init__()
        self.zip_pack: bool = False
        self.indent: Optional[int] = None

    def initialize(self, resources: Resources, configs: Config):
        super(JsonPackWriter, self).initialize(resources, configs)

        if not configs.output_dir:
            raise NotADirectoryError('Root output directory is not defined '
                                     'correctly in the configs.')

        if not os.path.exists(configs.output_dir):
            os.makedirs(configs.output_dir)

        self.zip_pack = configs.zip_pack
        self.indent = configs.indent

    @abstractmethod
    def sub_output_path(self, pack: DataPack) -> str:
        r"""Allow defining output path using the information of the pack.

        Args:
            pack: The input datapack.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        r"""This defines a basic ``Hparams`` structure.
        """
        config = super().default_configs()
        config.update({
            'output_dir': None,
            'zip_pack': False,
            'indent': None,
        })
        return config

    def _process(self, input_pack: DataPack):
        sub_path = self.sub_output_path(input_pack)
        if sub_path == '':
            raise ValueError(
                "No concrete path provided from sub_output_path.")

        maybe_create_dir(self.configs.output_dir)
        write_pack(input_pack, self.configs.output_dir, sub_path,
                   self.configs.indent, self.configs.zip_pack,
                   self.configs.overwrite)


class MultiPackWriter(BaseProcessor[MultiPack]):
    pack_base_out = 'packs'
    multi_base = 'multi'
    pack_idx = 'pack.idx'
    multi_idx = 'multi.idx'

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init
        super().initialize(resources, configs)

        pack_index = os.path.join(self.configs.output_dir, self.pack_idx)
        ensure_dir(pack_index)
        self.pack_idx_out = open(pack_index, 'w')

        multi_index = os.path.join(self.configs.output_dir, self.multi_idx)
        ensure_dir(multi_index)
        self.multi_idx_out = open(multi_index, 'w')

    def pack_name(self, pack: DataPack) -> str:
        r"""Allow defining output path using the information of the datapack.

        Args:
            pack: The input datapack.
        """
        return f"mult_pack_{pack.meta.pack_id}"

    def multipack_name(self, pack: MultiPack) -> str:
        r"""Allow defining output path using the information of the multipack.

        Args:
            pack: The input multipack.
        """
        return f"mult_pack_{pack.meta.pack_id}"

    def _process(self, input_pack: MultiPack):
        multi_out_dir = os.path.join(self.configs.output_dir, self.multi_base)
        pack_out_dir = os.path.join(self.configs.output_dir, self.pack_base_out)

        for pack in input_pack.packs:
            pack_out = write_pack(
                pack, pack_out_dir, self.pack_name(pack), self.configs.indent,
                self.configs.zip_pack, self.configs.overwrite)

            self.pack_idx_out.write(
                f'{pack.meta.pack_id}\t'
                f'{os.path.relpath(pack_out, self.configs.output_dir)}\n')

        multi_out = write_pack(
            input_pack, multi_out_dir,
            self.multipack_name(input_pack), self.configs.indent,
            self.configs.zip_pack, self.configs.overwrite
        )

        self.multi_idx_out.write(
            f'{input_pack.meta.pack_id}\t'
            f'{os.path.relpath(multi_out, self.configs.output_dir)}\n')

    def finish(self, _):
        self.pack_idx_out.close()
        self.multi_idx_out.close()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            'output_dir': None,
            'zip_pack': False,
            'indent': None,
        })
        return config
