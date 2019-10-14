from forte.data.base_pack import PackType
from forte.processors.base.writers import JsonPackWriter


class SimpleJsonPackWriter(JsonPackWriter):
    def _process(self, input_pack: PackType):
        pass

    def sub_output_path(self, pack: PackType) -> str:
        doc_id = pack.meta.doc_id
        if doc_id is None:
            raise ValueError(
                "Doc id for data pack not set, "
                "please supply to use this writer.")
        return doc_id
