# Copyright 2021 The Forte Authors. All Rights Reserved.
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

"""
TODO: Module docs
"""

import json
import logging
from typing import Dict, Any
import threading
import requests

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "RemoteProcessor"
]


class RemoteProcessor(PackProcessor):
    r"""
    TODO: Class Docs
    """
    def __init__(self):
        super().__init__()
        self._url: str
        self._barrier = threading.Barrier(2, timeout=10)

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self._url = f"http://{self.configs.host}:{self.configs.port}"
        response = requests.get(self._url)
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise ProcessorConfigError(f"{response.status_code}: "
                "Remote service not started or invalid endpoint configs.")
        logger.info("%s", response.json())

    def _process(self, input_pack: DataPack):
        response = requests.post(f"{self._url}/process", json={
            "args": [json.dumps([input_pack.serialize()])]
        })
        if response.status_code != 200:
            raise Exception(f"{response.status_code}: Invalid post request.")
        result = response.json()["result"]
        # TODO: Not recommended to directly update __dict__. Maybe it's better
        #   to add an "update()" interface to DataPack.
        input_pack.__dict__.update(DataPack.deserialize(result).__dict__)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        This defines a basic config structure for RemoteProcessor.
        Following are the keys for this dictionary:

            - ``port``: Port number for Stave server. Default value is `8888`.
            - ``host``: Host name for Stave server. Default value is
              `"localhost"`.

        Returns:
            dict: A dictionary with the default config for this processor.
        """
        config = super().default_configs()

        config.update({
            "port": 8008,
            "host": "localhost"
        })

        return config
