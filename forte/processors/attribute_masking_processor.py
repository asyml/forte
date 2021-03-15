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
from typing import Any, Dict

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "AttributeMasker"
]


class AttributeMasker(PackProcessor):
    def __init__(self):
        super().__init__()
        self.fields: Dict = {}

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resources: Resources, config: Config):
        super().initialize(resources, config)
        self.fields = config.kwargs

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""Default config for this processor.

        Example usage is shown below

        .. code-block:: python
            {
                "kwargs": {
                    Token: ["ner"]
                }
            }

        Here:

        `"request"`: dict
            The entry types and fields required. The keys of the requests dict
            are the entry types whose fields need to be masked and the value is
            a list of field names.
        """
        config = super().default_configs()
        config.update({
            "type": "",
            "kwargs": {}
        })
        return config

    def _process(self, input_pack: DataPack):
        for entry_type, attributes in self.fields.items():
            for entry in input_pack.get_entries_of(entry_type):
                for attribute in attributes:
                    setattr(entry, attribute, None)
