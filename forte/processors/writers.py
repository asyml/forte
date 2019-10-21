from forte.data.base_pack import PackType
from forte.processors.base.writers import JsonPackWriter


class DocIdJsonPackWriter(JsonPackWriter[PackType]):
    # pylint: disable=no-self-use
    def sub_output_path(self, pack: PackType) -> str:
        if pack.meta.doc_id is None:
            raise ValueError(
                "Cannot use DocIdJsonPackWriter when doc id is not set.")
        return pack.meta.doc_id + '.json'
