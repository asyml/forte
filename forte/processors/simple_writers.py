from texar.torch import HParams

from forte.common import Resources
from forte.common.types import PackType
from forte.processors.base.writers import JsonPackWriter


class SimpleJsonPackWriter(JsonPackWriter):
    def initialize(self, configs: HParams, resource: Resources):
        return super().initialize(configs, resource)

    def _process(self, input_pack: PackType):
        pass

    def sub_output_dir(self, pack: PackType) -> str:
        return ''
