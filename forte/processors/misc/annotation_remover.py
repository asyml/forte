from typing import Any, Dict

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.processors.base import PackProcessor
from forte.utils import get_class

__all__ = ["AnnotationRemover"]


class AnnotationRemover(PackProcessor):
    def _process(self, input_pack: DataPack):
        for type_name in self.configs.removal_types:
            type_cls = get_class(type_name)

            # Note: we cannot delete during iteration, which may cause
            # un-expected behavior in the iterator.
            t: Entry
            for t in list(input_pack.get(type_cls)):
                input_pack.delete_entry(t)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({"removal_types": []})
        return config
