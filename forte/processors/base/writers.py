"""
Writers are simply processors with the side-effect to write to the disk.
This file provide some basic writer implementations.
"""

import os
import logging
from abc import abstractmethod, ABC
import gzip

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import PackType
from forte.processors import ProcessInfo
from forte.processors.base.base_processor import BaseProcessor
from forte.data.io_utils import ensure_dir

logger = logging.getLogger(__name__)

__all__ = [
    'JsonPackWriter',
]


class JsonPackWriter(BaseProcessor, ABC):
    def __init__(self):
        super().__init__()
        self.root_output_dir: str = ''
        self.zip_pack: bool = False

    def initialize(self, configs: HParams, resource: Resources):
        self.root_output_dir = configs.output_dir
        self.zip_pack = configs.zip_pack

        if not self.root_output_dir:
            raise NotADirectoryError('Root output directory is not defined '
                                     'correctly in the configs.')

        if not os.path.exists(self.root_output_dir):
            os.makedirs(self.root_output_dir)

    def _define_input_info(self) -> ProcessInfo:
        # No specific requirements from the writer, it can write anything to
        # DataPack.
        return {}

    def _define_output_info(self) -> ProcessInfo:
        # This writer don't create any entries.
        return {}

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
