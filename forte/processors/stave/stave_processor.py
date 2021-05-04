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
pipeline result. It supports serving Stave instance with tornado(for static
files) and django(for web apis). Forte users can plug it into the pipeline
to easily visualize datapacks with annotations. It is also highly
configurable for users to change port number, layout, ontology, etc.

Package Requirements:
    tornado == 6.1
    django == 3.2
    forte
    *stave (Required in future version)

Required environment variables:
*All of the variables below will be deprecated in future version.
    FRONTEND_BUILD_PATH:
        Absolute path (or relative path from PYTHONPATH)
        to stave build folder. Example: "stave/build/"
    DJANGO_BACKEND_PATH:
        Absolute path (or relative path from PYTHONPATH)
        to django backend folder. Example: "stave/simple-backend/"
"""

import os
import sys
import json
import logging
import asyncio
import threading
import collections
import webbrowser
from typing import Dict, Set, Any
import requests

import django
from django.core.wsgi import get_wsgi_application

from tornado.web import FallbackHandler, StaticFileHandler, \
    RequestHandler, url, Application
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.ontology.code_generation_objects import search
from forte.processors.base import PackProcessor
from forte.common.exception import ProcessExecutionException

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
    It then genertes default configurations based on the input
    ontology to start a full-fledged stave instance without any additional
    specification by users.
    Example usage:
        pipeline.add(StaveProcessor())

    `StaveProcessor` is also highly customizable for users to set up. Users may
    configure port number, server host, project name, etc.
    Example usage:
        pipeline.add(StaveProcessor(), configs={
            "port": 8880,
            "projectName": "serialization_pipeline_test"
        })

    `StaveProcessor` automatically creates project and documents via stave
    backend api based on project configurations and input datapacks.
    If users would like to modify project configs, it can pass it to the
    processor by changing `projectConfigs` field.
    Example usage:
        pipeline.add(StaveProcessor(), configs={
            "port": 8879,
            "projectConfigs": {
                # Configure Stave layout
                "layoutConfigs": {
                    "center-middle": "DialogueBox"
                }
            }
        })
    """

    def __init__(self):
        super().__init__()
        self._url: str
        self._server_started: bool = False
        self._project_id: int = -1

        # Used for sync between threads
        self._barrier = threading.Barrier(2, timeout=1)

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self._url = f"http://{self.configs.host}:{self.configs.port}"

        # Generate default project configurations
        try:
            self.configs.projectConfigs = Config(
                hparams=self.configs.projectConfigs,
                default_hparams=self._default_project_configs()
            )
        except Exception as e:
            raise ProcessorConfigError(
                "`projectConfig` not correctly set.") from e

        # Validate multi_pack project config:
        #   A `multi_pack` project must have `multiOntology` set.
        if self.configs.projectType != "single_pack" and (
            self.configs.projectType != "multi_pack" or
            self.configs.multiOntology is None):
            raise ProcessorConfigError("Invalid project type configuration.")

    class ProxyHandler(FallbackHandler):
        """
        URL routing for django web interface.
        ProxyHandler directs all requests with `/api`-prefixed url to
        the django wsgi application.
        """

        def initialize(self, fallback):
            # Strip prefix "/api" from uri and path
            self.request.uri = self.request.uri[4:]
            self.request.path = self.request.path[4:]
            super().initialize(fallback)

    class ReactHandler(RequestHandler):
        """
        Handler of requests to React index page
        ReactHandler makes sure all requests fall back to index page
        so that they can follow the standard React routing rules.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._build_path: str

        def initialize(self, build_path):
            self._build_path = build_path

        def get(self):
            self.render(self._build_path + "/index.html")

    def _process(self, input_pack: DataPack):
        if not self._server_started:
            # Start a new thread to server Stave
            thread = threading.Thread(
                target=self._start_server,
                daemon=self.configs.server_thread_daemon
            )
            thread.start()

            # Wait for server to boot up
            self._barrier.wait()

            if not thread.is_alive():
                raise ProcessExecutionException(
                    "%s: Stave server not started." % self.configs.projectName)
            self._server_started = True

        if self._server_started:
            self._create_project(input_pack)

    def _start_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        sys.path.insert(0, self.configs.django_backend_path)
        os.environ['DJANGO_SETTINGS_MODULE'] = "nlpviewer_backend.settings"
        os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
        django.setup()

        wsgi_app = WSGIContainer(get_wsgi_application())
        build_path = self.configs.frontend_build_path

        # TODO: Find better logic to deal with routing.
        #       The following implementation may miss some corner cases.
        app = Application([
            url(r"/api/(.*)", self.ProxyHandler, dict(fallback=wsgi_app)),
            url(r'.*/static/(.*)', StaticFileHandler, {
                "path": build_path + "/static/"
            }),
            url(r'.*/([^/]*\.png)', StaticFileHandler, {"path": build_path}),
            url(r'.*/([^/]*\.ico)', StaticFileHandler, {"path": build_path}),
            url(r"/.*", self.ReactHandler, {"build_path": build_path})
        ])

        server = HTTPServer(app)
        server.listen(self.configs.port)

        # Release lock in main thread
        self._barrier.wait()

        IOLoop.current().start()

    def _create_project(self, input_pack: DataPack):
        with requests.Session() as session:

            # Log in as admin user
            response = session.post(f"{self._url}/api/login",
                json={
                    "name": self.configs.user_name,
                    "password": self.configs.user_password
                })
            logger.info("%d %s", response.status_code, response.text)
            if response.status_code != 200:
                return

            # Configure and create project
            if self._project_id < 0:
                response = session.post(f"{self._url}/api/projects/new",
                    json={
                        "type": self.configs.projectType,
                        "name": self.configs.projectName,
                        "ontology": json.dumps(
                            self.resources.get("onto_specs_dict")
                        ),
                        "multiOntology": str(self.configs.multiOntology),
                        "config": str(self.configs.projectConfigs)
                    })
                logger.info("%d %s", response.status_code, response.text)
                if response.status_code != 200:
                    return
                self._project_id = response.json()["id"]
                webbrowser.open(self._url)

            # Configure and create document
            response = session.post(f"{self._url}/api/documents/new",
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
            - frontend_build_path: Absolute path (or relative path
                from PYTHONPATH) to stave build folder.
                Example: "stave/build/".
                default value set from env var FRONTEND_BUILD_PATH.
            - django_backend_path: Absolute path (or relative path
                from PYTHONPATH) to django backend folder.
                Example: "stave/simple-backend/".
                default value set from env var DJANGO_BACKEND_PATH.
            - port: port number for Stave server. default value is 8888.
            - host: host name for Stave server. default value is `localhost`.
            - user_name: admin user name. default to `admin`.
            - user_password: admin user password. default to `admin`.
            - projectType: single_pack(default) / multi_pack.
            - projectName: project name displayed on Stave.
                default name is `Auto generated project`.
            - multiOntology: a dictionary for multi_pack ontology
                default to {}.
            - projectConfigs: project configurations.
                default value generated from ontology.
            - server_thread_daemon: sets whether the thread is daemonic.
                default to False.
        """
        config = super().default_configs()

        config.update({
            "frontend_build_path": os.environ["FRONTEND_BUILD_PATH"],
            "django_backend_path": os.environ["DJANGO_BACKEND_PATH"],
            "port": 8888,
            "host": "localhost",
            "user_name": "admin",
            "user_password": "admin",
            "projectType": "single_pack",
            "projectName": "Auto generated project",
            "multiOntology": None,
            "projectConfigs": None,
            "server_thread_daemon": False
        })

        return config
