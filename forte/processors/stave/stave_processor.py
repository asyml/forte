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
A StaveProcessor is introduced to enable immediate visualization of forte
pipeline result. It supports serving Stave instance with StaveViewer.
Forte users can plug it into the pipeline to easily visualize datapacks
with annotations. It is also highly configurable for users to change port
number, host name, layout, etc.

Package Requirements:
    forte
    *stave (Required in future version)

Required environment variables:
*All of the variables below will be deprecated in future version.
    FRONTEND_BUILD_PATH:
        Absolute path (or relative path from PYTHONPATH)
        to stave build folder. Example: "stave/build/"
    PYTHONPATH:
        Absolute path to django backend folder (e.g.,
        "$STAVE_PATH/simple-backend/") should be inserted into PYTHONPATH.
"""

import os
import json
import logging
import collections
from typing import Dict, Set, Any
import requests

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.ontology.code_generation_objects import search
from forte.processors.base import PackProcessor

from nlpviewer_backend.lib.stave_viewer import StaveViewer
from nlpviewer_backend.lib.stave_project import StaveProjectWriter

logger = logging.getLogger(__name__)

__all__ = [
    "StaveProcessor"
]


class StaveProcessor(PackProcessor):
    """
    `StaveProcessor` provides easy visualization for forte users. We can
    visualize datapack with annotations by inserting it into the forte
    pipeline without affecting the original functionalities.

    `StaveProcessor` requires an ontology file being passed to the pipeline.
    It then genertes default configurations based on the input ontology to
    start a stave instance without any additional specification by users.
    Example usage:
        pipeline.add(StaveProcessor())

    `StaveProcessor` is also highly customizable for users to set up. Users may
    configure port number, server host, project name, etc.
    Example usage:
        pipeline.add(StaveProcessor(), configs={
            "port": 8880,
            "project_name": "serialization_pipeline_test"
        })

    Users can modify project configs by changing the `project_configs` field.
    Example usage:
        pipeline.add(StaveProcessor(), configs={
            "port": 8879,
            "project_configs": {
                # Configure Stave layout
                "layoutConfigs": {
                    "center-middle": "DialogueBox"
                }
            }
        })
    """

    def __init__(self):
        super().__init__()
        self._project_id: int = -1
        self._viewer: StaveViewer
        self._project_writer: StaveProjectWriter

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # Validate multi_pack project config:
        #   A `multi_pack` project must have `multi_ontology` set.
        if self.configs.project_type != "single_pack" and (
            self.configs.project_type != "multi_pack" or
            self.configs.multi_ontology is None):
            raise ProcessorConfigError("Invalid project type configuration.")

        # Generate default configurations and write to project folder
        try:
            self.configs.project_configs = Config(
                hparams=self.configs.project_configs,
                default_hparams=self._default_project_configs()
            )
            self.configs.multi_ontology = \
                self.configs.multi_ontology or Config({}, {})
            self.configs.project_path = os.path.abspath(
                self.configs.project_path or self.configs.project_name)

            self._viewer = StaveViewer(
                build_path=os.environ["FRONTEND_BUILD_PATH"],
                project_path=self.configs.project_path,
                host=self.configs.host,
                port=self.configs.port,
                thread_daemon=self.configs.server_thread_daemon,
                in_viewer_mode=self.configs.in_viewer_mode
            )

            if self._viewer.in_viewer_mode:
                self._project_writer = StaveProjectWriter(
                    project_path=self.configs.project_path,
                    project_name=self.configs.project_name,
                    project_type=self.configs.project_type,
                    ontology=self.resources.get("onto_specs_dict"),
                    project_configs=self.configs.project_configs.todict(),
                    multi_ontology=self.configs.multi_ontology.todict()
                )

        except Exception as e:
            raise ProcessorConfigError(
                "Invalid conifguration for StaveProcessor.") from e

    def _process(self, input_pack: DataPack):

        if not self._viewer.server_started:
            self._viewer.run()

        if self._viewer.server_started:
            if self._viewer.in_viewer_mode:
                textpack_id = self._project_writer.write_textpack(
                    input_pack.pack_name
                        if self.configs.use_pack_name
                        else input_pack.pack_id,
                    input_pack.serialize()
                )
                if textpack_id == 0:
                    self._viewer.open()
            else:
                self._create_document(input_pack)

    def _create_document(self, input_pack: DataPack):
        with requests.Session() as session:

            # Log in as admin user
            response = session.post(f"{self._viewer.url}/api/login",
                json={
                    "name": self.configs.user_name,
                    "password": self.configs.user_password
                })
            logger.info("%d %s", response.status_code, response.text)
            if response.status_code != 200:
                return

            # Configure and create project
            if self._project_id < 0:
                response = session.post(f"{self._viewer.url}/api/projects/new",
                    json={
                        "type": self.configs.project_type,
                        "name": self.configs.project_name,
                        "ontology": json.dumps(
                            self.resources.get("onto_specs_dict")
                        ),
                        "multiOntology": str(self.configs.multi_ontology),
                        "config": str(self.configs.project_configs)
                    })
                logger.info("%d %s", response.status_code, response.text)
                if response.status_code != 200:
                    return
                self._project_id = response.json()["id"]
                self._viewer.open()

            # Configure and create document
            response = session.post(f"{self._viewer.url}/api/documents/new",
                json={
                    "name": input_pack.pack_name,
                    "textPack": input_pack.serialize(),
                    "project_id": self._project_id,
                })
            logger.info("%d %s", response.status_code, response.text)
            if response.status_code != 200:
                return

    def _default_project_configs(self):
        # pylint: disable=line-too-long
        """
        Create default project configuration based on ontology.
        This is translated from JavaScript function `createDefaultConfig` in
        https://github.com/asyml/stave/blob/d82383de3d74bf09c0d30f33d8a902595f5aff80/src/app/pages/Projects.tsx#L140

        Returns:
            configs: A dictionary with the default config for project.

        """
        # pylint: enable=line-too-long

        try:
            ontology = self.resources.get("onto_specs_dict")
            entry_tree = self.resources.get("merged_entry_tree")
        except Exception as e:
            raise ProcessorConfigError(
                "onto_specs_dict/merged_entry_tree is"
                " not set in resources.") from e

        configs: Dict[str, Any] = {
            "legendConfigs": {},
            "scopeConfigs": {},
            "layoutConfigs": {
                "center-middle": "default-nlp",
                "left": "default-meta",
                "right": "default-attribute",
                "center-bottom": "disable"
            }
        }

        # Create legend configs
        legend_configs: Dict[str, Any] = {}
        entry_name_set: Set[str] = set()
        for entry in ontology["definitions"]:
            entry_name = entry["entry_name"]
            entry_name_set.add(entry_name)
            legend_configs[entry_name] = {
                "is_selected": False,
                "is_shown": True,
            }
            if "attributes" in entry and len(entry["attributes"]) > 0:
                attributes_configs = {}
                for attribute in entry["attributes"]:
                    if attribute["type"] == "str":
                        attributes_configs[attribute["name"]] = False
                legend_configs[entry_name]["attributes"] = attributes_configs
        configs["legendConfigs"] = legend_configs

        # Find all subclass of `forte.data.ontology.top.Annotation` and
        # update `scopeConfigs` accordingly.
        queue = collections.deque([
            search(entry_tree.root, "forte.data.ontology.top.Annotation")
        ])
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.pop()
                if node.name in entry_name_set:
                    configs['scopeConfigs'][node.name] = False
                for entry in node.children:
                    queue.appendleft(entry)
        return configs

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        This defines a basic config structure for StaveProcessor.
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - build_path: Absolute path to stave build folder.
                Example: "$STAVE_PATH/build/".
            - port: port number for Stave server. default value is 8888.
            - host: host name for Stave server. default value is `localhost`.
            - user_name: admin user name. default to `admin`.
            - user_password: admin user password. default to `admin`.
            - project_type: single_pack(default) / multi_pack.
            - project_name: project name displayed on Stave.
                default name is `Auto generated project`.
            - multi_ontology: a dictionary for multi_pack ontology
                default to {}.
            - project_configs: project configurations.
                default value generated from ontology.
            - server_thread_daemon: sets whether the thread is daemonic.
                default to False.
            - in_viewer_mode: Enable viewer mode of Stave. If False,
                StaveViewer will start a standard Stave instance.
                Default to True.
            - use_pack_name: Use `pack_name` to name the textpack being saved
                to project path in viewer mode. If False, will use `pack_id`
                for naming. Default to False.
        """
        config = super().default_configs()

        config.update({
            "project_path": None,
            "port": 8888,
            "host": "localhost",
            "user_name": "admin",
            "user_password": "admin",
            "project_type": "single_pack",
            "project_name": "Auto generated project",
            "multi_ontology": None,
            "project_configs": None,
            "server_thread_daemon": False,
            "in_viewer_mode": True,
            "use_pack_name": False
        })

        return config
