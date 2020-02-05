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

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor


__all__ = [
    "AttributeMasker"
]


class AttributeMasker(PackProcessor):

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, _: Resources, config: HParams):
        self.fields = config.kwargs

    @staticmethod
    def default_configs() -> Dict[str, Any]:
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
        return {
            "type": "",
            "kwargs": {}
        }

    def _process(self, input_pack: DataPack):
        for entry_type, attributes in self.fields:
            for entry in input_pack.get_entries_by_type(entry_type):
                entry.set_fields(
                    **{attribute: None for attribute in attributes})
