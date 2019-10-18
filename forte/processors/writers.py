from forte.data import DataPack
from forte.processors.base.writers import JsonPackWriter


class DocIdJsonPackWriter(JsonPackWriter):
    def sub_output_path(self, pack: DataPack) -> str:
        if pack.meta.doc_id is None:
            raise ValueError(
                "Cannot use DocIdJsonPackWriter when doc id is not set.")
        return pack.meta.doc_id + '.json'
