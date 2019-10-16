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
from forte.data.io_utils import ensure_dir
from forte.processors.base.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

__all__ = [
    'JsonPackWriter',
]


class JsonPackWriter(BaseProcessor, ABC):
    def __init__(self):
        super().__init__()
        self.root_output_dir: str = ''
        self.zip_pack: bool = False

    def initialize(self, resource: Resources, configs: HParams):
        self.root_output_dir = configs.output_dir
        self.zip_pack = configs.zip_pack

        if not self.root_output_dir:
            raise NotADirectoryError('Root output directory is not defined '
                                     'correctly in the configs.')

        if not os.path.exists(self.root_output_dir):
            os.makedirs(self.root_output_dir)

    @abstractmethod
    def sub_output_dir(self, pack: PackType) -> str:
        """
        Allow defining output path using the information of the pack.
        Args:
            pack:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        return {
            'output_dir': None,
            'zip_pack': True,
        }

    def _process(self, input_pack: PackType):
        p = os.path.join(self.root_output_dir, self.sub_output_dir(input_pack))
        ensure_dir(p)

        if self.zip_pack:
            with gzip.open(p + '.gz', 'wt') as out:
                out.write(input_pack.serialize())
        else:
            with open(p, 'w') as out:
                out.write(input_pack.serialize())
