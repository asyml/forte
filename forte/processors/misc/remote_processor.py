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
from typing import Dict, Set, Any, Optional

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.utils import create_import_error_msg


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
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                create_import_error_msg(
                    "requests", "remote", "Remote Processor"
                )
            ) from e
        self._requests: Any = requests
        self._records: Optional[Dict[str, Set[str]]] = None
        self._expectation: Optional[Dict[str, Set[str]]] = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        _validation: Config = self.configs.validation

        # Verify the service is running
        response = self._requests.get(self.configs.url)
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise ProcessorConfigError(
                f"{response.status_code} {response.reason}: Please double "
                "check your endpoint URL configuration and make sure that the "
                f"remote service at {self.configs.url} is a valid pipeline "
                "service that is up and running."
            )
        service_name: str = response.json()["service_name"]
        input_format: str = response.json()["input_format"]

        if _validation.do_init_type_check:
            # Validate service name and input format
            if service_name != _validation.expected_name:
                raise ProcessorConfigError(
                    "Validation fail: The expected service name "
                    f"('{_validation.expected_name}') does not match the "
                    "actual name returned by remote service "
                    f"('{service_name}'). Please double check your endpoint "
                    f"URL {self.configs.url} or consider updating the configs "
                    "of RemoteProcessor so that 'validation.expected_name' "
                    f"equals to '{service_name}'."
                )
            if input_format != _validation.input_format:
                raise ProcessorConfigError(
                    "Validation fail: The expected input format "
                    f"('{_validation.input_format}') does not match the "
                    "actual input format returned by remote service "
                    f"('{input_format}'). Please double check your endpoint "
                    f"URL {self.configs.url} or consider updating the configs "
                    "of RemoteProcessor so that 'validation.input_format' "
                    f"equals to '{input_format}'."
                )

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `RemoteProcessor`. The records
        are queried from the remote service. The types and attributes are
        populated from all the components in remote pipeline.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        if self._records is None:
            response = self._requests.get(f"{self.configs.url}/records")
            if response.status_code != 200 or response.json()["status"] != "OK":
                raise ProcessorConfigError(
                    f"{response.status_code} {response.reason}: "
                    "Fail to fetch records from remote service. Please make "
                    f"sure that the remote service at {self.configs.url} is "
                    "a valid pipeline service that is up and running."
                )
            self._records = response.json()["records"]
        record_meta.update(self._records)

    def expected_types_and_attributes(self):
        r"""Method to add expected types and attributes for the input of
        `RemoteProcessor`. This should be the `expected_types_and_attributes`
        of the first processor in remote pipeline.
        """
        if self._expectation is None:
            response = self._requests.get(f"{self.configs.url}/expectation")
            if response.status_code != 200 or response.json()["status"] != "OK":
                raise ProcessorConfigError(
                    f"{response.status_code} {response.reason}: "
                    "Fail to fetch expected types and attributes from remote "
                    "service. Please make sure that the remote service at "
                    f"{self.configs.url} is a valid pipeline service that is "
                    "up and running."
                )
            self._expectation = response.json()["expectation"]
        return self._expectation

    def _process(self, input_pack: DataPack):
        # Pack the input_pack and POST it to remote service
        response = self._requests.post(
            f"{self.configs.url}/process",
            json={"args": json.dumps([[input_pack.to_string()]])},
        )
        if response.status_code != 200 or response.json()["status"] != "OK":
            raise Exception(
                f"{response.status_code} {response.reason}: "
                "Invalid post request to process input pack. Please make "
                f"sure that the remote service at {self.configs.url} is "
                "a valid pipeline service that is up and running."
            )
        result = response.json()["result"]
        input_pack.update(DataPack.from_string(result))  # type: ignore

    def set_test_mode(self, app):
        """
        Configure the processor into test mode. This should only be called
        from a pytest program.

        Args:
            app: A fastapi app from a Forte pipeline.
        """
        try:
            # pylint: disable=import-outside-toplevel
            from fastapi.testclient import TestClient
        except ImportError as err:
            raise ImportError(
                create_import_error_msg("fastapi", "remote", "RemoteProcessor")
            ) from err
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

        Returns:
            dict: A dictionary with the default config for this processor.
        """
        return {
            "url": "http://localhost:8008",
            "validation": {
                "do_init_type_check": False,
                "input_format": "string",
                "expected_name": "",
            },
        }
