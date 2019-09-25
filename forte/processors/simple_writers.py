from forte.data.base_pack import PackType
from forte.processors.base.writers import JsonPackWriter


class SimpleJsonPackWriter(JsonPackWriter):
    def _process(self, input_pack: PackType):
        pass

    def sub_output_dir(self, pack: PackType) -> str:
        return ''
