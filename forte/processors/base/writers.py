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
import logging
import os
from abc import abstractmethod, ABC

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.processors.base.base_processor import BaseProcessor
from forte.utils.utils_io import maybe_create_dir, ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    'JsonPackWriter',
]


class JsonPackWriter(BaseProcessor[PackType], ABC):
    def __init__(self):
        super().__init__()
        self.root_output_dir: str = ''
        self.zip_pack: bool = False

    def initialize(self, _: Resources, configs: HParams):
        self.root_output_dir = configs.output_dir
        self.zip_pack = configs.zip_pack

        if not self.root_output_dir:
            raise NotADirectoryError('Root output directory is not defined '
                                     'correctly in the configs.')

        if not os.path.exists(self.root_output_dir):
            os.makedirs(self.root_output_dir)

    @abstractmethod
    def sub_output_path(self, pack: PackType) -> str:
        r"""Allow defining output path using the information of the pack.

        Args:
            pack: The input datapack.
        """
        raise NotImplementedError

    @staticmethod
    def default_configs():
        r"""This defines a basic ``Hparams`` structure.
        """
        return {
            'output_dir': None,
            'zip_pack': False,
        }

    def _process(self, input_pack: PackType):
        sub_path = self.sub_output_path(input_pack)
        if sub_path == '':
            raise ValueError(
                "No concrete path provided from sub_output_path.")

        maybe_create_dir(self.root_output_dir)
        p = os.path.join(self.root_output_dir, sub_path)

        ensure_dir(p)

        if self.zip_pack:
            with gzip.open(p + '.gz', 'wt') as out:
                out.write(input_pack.serialize())
        else:
            with open(p, 'w') as out:
                out.write(input_pack.serialize())
