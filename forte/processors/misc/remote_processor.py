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
from typing import Dict, Set, Any
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
        self._validation: Config
        self._records: Dict[str, Set[str]]

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self._validation = self.configs.validation

        # Verify the service is running
        response = self._requests.get(self.configs.url)
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise ProcessorConfigError(
                f"{response.status_code} {response.reason}: "
                "Remote service not started or invalid endpoint configs."
            )
        service_name: str = response.json()["service_name"]
        input_format: str = response.json()["input_format"]

        if self._validation.do_init_type_check:
            # Validate service name and input format
            if service_name != self._validation.expected_name:
                raise ProcessorConfigError(
                    "Validation fail: The expected service name "
                    f"('{self._validation.expected_name}') does not match "
                    "the actual name returned by remote service "
                    f"('{service_name}'). Consider updating the configs of "
                    "RemoteProcessor so that 'validation.expected_name' "
                    f"equals to '{service_name}'."
                )
            if input_format != self._validation.input_format:
                raise ProcessorConfigError(
                    "Validation fail: The expected input format "
                    f"('{self._validation.input_format}') does not match "
                    "the actual input format returned by remote service "
                    f"('{input_format}'). Consider updating the configs of "
                    "RemoteProcessor so that 'validation.input_format' "
                    f"equals to '{input_format}'."
                )

            # Get the output records
            response = self._requests.get(f"{self.configs.url}/records")
            if response.status_code != 200 or response.json()["status"] != "OK":
                raise ProcessorConfigError(
                    f"{response.status_code} {response.reason}: "
                    "Fail to fetch records from remote service."
                )
            self._records = response.json()["records"]

    def check_record(self, input_pack):
        r"""Add additional checking of remote pipeline service. The records
        of remote service will be queried and validated against the expected
        record types and attributes in configs.

        Args:
            input_pack: The input datapack.
        """
        super().check_record(input_pack)
        if self._validation.do_init_type_check:
            # Validate the output records
            record_types_and_attributes_check(
                self._validation.expected_records.todict(), self._records
            )

    def _process(self, input_pack: DataPack):
        # Pack the input_pack and POST it to remote service
        response = self._requests.post(
            f"{self.configs.url}/process",
            json={"args": json.dumps([[input_pack.serialize()]])},
        )
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise Exception(
                f"{response.status_code} {response.reason}: "
                "Invalid post request."
            )
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
            - ``validation``: Information for validation.

                - ``do_init_type_check``: Validate the pipeline by checking
                  the info of the remote pipeline with the expected
                  attributes. Default to `False`.
                - ``input_format``: The expected input format of the remote
                  service. Default to `"string"`.
                - ``expected_name``: The expected pipeline name.
                  Default to `''`.
                - ``expected_records``: The expected records of the output
                  DataPack meta from the pipeline. It should be a string that
                  represents a serialized dictionary `Dict[str, Set[str]]`.
                  Default to `None`.

        Returns:
            dict: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "url": "http://localhost:8008",
                "validation": {
                    "do_init_type_check": False,
                    "input_format": "string",
                    "expected_name": "",
                    "expected_records": None,
                },
            }
        )
        return config
