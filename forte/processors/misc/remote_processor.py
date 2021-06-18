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
RemoteProcessor is used to interact with a remote Forte end-point.
The Forte service must be created by a pipeline with `RawDataDeserializeReader`
being set as its reader.
"""

import json
import logging
from typing import Dict, Any
import requests

from fastapi import FastAPI
from fastapi.testclient import TestClient

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.utils.utils_processor import record_types_and_attributes_check

logger = logging.getLogger(__name__)

__all__ = ["RemoteProcessor"]


class RemoteProcessor(PackProcessor):
    r"""
    RemoteProcessor wraps up the interactions with remote Forte end point.
    Each input DataPack from the upstream component will be serialized and
    packed into a POST request to be sent to a remote service, which should
    return a response that can be parsed into a DataPack to update the input.
    Example usage:

    .. code-block:: python

        # Assume that a Forte service is running on "localhost:8080".
        Pipeline() \
            .set_reader(plaintext_reader(), {"input_path":"some/path"}) \
            .add(RemoteProcessor(), {"url": "http://localhost:8008"})

    """

    def __init__(self):
        super().__init__()
        self._requests: Any = requests

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # Verify the service is running
        response = self._requests.get(self.configs.url)
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise ProcessorConfigError(
                f"{response.status_code}: "
                "Remote service not started or invalid endpoint configs."
            )
        service_name: str = response.json()["name"]

        if self.configs.do_validation:
            # Validate service name
            if service_name != self.configs.expected_name:
                raise ProcessorConfigError(
                    "Validation fail: Name Mismatch. "
                    f"'{service_name}' != '{self.configs.expected_name}'"
                )

            # Validate the output records
            response = self._requests.get(f"{self.configs.url}/records")
            if response.status_code != 200 or response.json()["status"] != "OK":
                raise ProcessorConfigError(
                    f"{response.status_code}: "
                    "Fail to fetch records from remote service."
                )
            records = {
                k: set(v)
                for k, v in json.loads(response.json()["records"]).items()
            }
            expectation = {
                k: set(v)
                for k, v in json.loads(self.configs.expected_records).items()
            }
            record_types_and_attributes_check(expectation, records)

    def _process(self, input_pack: DataPack):
        # Pack the input_pack and POST it to remote service
        response = self._requests.post(
            f"{self.configs.url}/process",
            json={"args": json.dumps([[input_pack.serialize()]])},
        )
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise Exception(f"{response.status_code}: Invalid post request.")
        result = response.json()["result"]
        input_pack.update(DataPack.deserialize(result))

    def set_test_mode(self, app: FastAPI):
        """
        Configure the processor into test mode. This should only be called
        from a pytest program.

        Args:
            app: A fastapi app from a Forte pipeline.
        """
        self._requests = TestClient(app)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        This defines a basic config structure for RemoteProcessor.
        Following are the keys for this dictionary:

            - ``url``: URL of the remote service end point.
              Default value is `"http://localhost:8008"`.
            - ``do_validation``: Validate the pipeline by checking the info
              of the remote pipeline with the expected attributes. Default
              to `False`.
            - ``expected_name``: The expected pipeline name. Default to `''`.
            - ``expected_records``: The expected records of the output
              DataPack meta from the pipeline. It should be a string that
              represents a serialized dictionary `Dict[str, List[str]]`.
              Default to `{}`.

        Returns:
            dict: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "url": "http://localhost:8008",
                "do_validation": False,
                "expected_name": "",
                "expected_records": "{}",
            }
        )
        return config
