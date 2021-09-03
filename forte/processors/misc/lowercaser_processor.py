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
import logging
from typing import Dict, Any

from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "LowerCaserProcessor",
]


class LowerCaserProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        text = input_pack.text
        for src, tgt in self.configs.custom_substitutions:
            text = text.replace(src, tgt)

        lower_text = text.lower()
        if len(lower_text) == len(text):
            input_pack.set_text(lower_text.lower())
        else:
            error_char = ""
            for c in text:
                if not c.lower().upper() == c:
                    error_char = c
                    break

            logging.error(
                "Some characters cannot be converted to lower case without "
                "changing length in pack [%s] will "
                "result in a change of text length, which will cause "
                "problems in the data pack system. The text of this pack "
                "will remain unchanged. One way to solve this is to provide "
                "values from the 'custom_substitutions'. The first "
                "problematic character is [%s].",
                input_pack.pack_id,
                error_char,
            )

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Default configurations for this processor, it contains the following
        configuration values:

        - "custom_substitutions": a dictionary contains the mapping
            used to conduct lower case, {"İ": "i"}. The length (`len`) of the
            two string must be the same.

        Returns:

        """
        return {
            "custom_substitutions": {},
            "@no_typecheck": ["custom_substitutions"],
        }
