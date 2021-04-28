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

'''
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
    ONTOLOGY_PATH:
        Absolute path to the ontology file.
    DJANGO_BACKEND_PATH:
        Absolute path (or relative path from PYTHONPATH)
        to django backend folder. Example: "stave/simple-backend/"
'''

import os
import sys
import json
import time
import logging
import asyncio
import threading
import collections
import webbrowser
from typing import Dict, Any
import requests

import django
from django.core.wsgi import get_wsgi_application

from tornado.web import FallbackHandler, StaticFileHandler, \
    RequestHandler, url, Application
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.common.exception import ProcessExecutionException

logger = logging.getLogger(__name__)

__all__ = [
    "StaveProcessor"
]


class StaveProcessor(PackProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._url: str
        self._server_started: bool = False
        self._project_id: int = -1

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self._url = f"http://{self.configs.host}:{self.configs.port}"

    class ProxyHandler(FallbackHandler):
        '''
        URL routing for django web interface
        '''

        def initialize(self, fallback):
            # Strip prefix "/api" from uri and path
            self.request.uri = self.request.uri[4:]
            self.request.path = self.request.path[4:]
            super().initialize(fallback)

    class ReactHandler(RequestHandler):
        '''
        Handler of requests to React index page
        '''

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
                daemon=False
            )
            thread.start()

            # Wait for server to boot up
            time.sleep(0.5)

            if not thread.is_alive():
                raise ProcessExecutionException(
                    "%s: Stave server not started." % self.configs.projectName)
            webbrowser.open(self._url)
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
                        "ontology": self.configs.ontology.__str__(),
                        "multiOntology": self.configs.multiOntology.__str__(),
                        "config": self.configs.projectConfigs.__str__()
                    })
                logger.info("%d %s", response.status_code, response.text)
                if response.status_code != 200:
                    return
                self._project_id = response.json()["id"]

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

    @classmethod
    def _default_project_configs(cls, ontology: Dict[str, Any]):
        """
        Create default project configuration based on ontology.
        This is translated from JavaScript function `createDefaultConfig` in
        asyml/stave/src/app/pages/Projects.tsx

        Args:
            ontology: A dictionary representing ontology.

        Returns:
            configs: A dictionary with the default config for project.

        """
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

        legend_configs: Dict[str, Any] = {}
        entry_graph = collections.defaultdict(set)
        annotation_entry = "forte.data.ontology.top.Annotation"
        for entry in ontology["definitions"]:
            entry_name = entry["entry_name"]

            if entry_name == annotation_entry:
                configs['scopeConfigs'][entry_name] = False
            else:
                entry_graph[entry["parent_entry"]].add(entry_name)

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

        children = collections.deque(entry_graph[annotation_entry])
        while children:
            size = len(children)
            for _ in range(size):
                child = children.pop()
                configs['scopeConfigs'][child] = False
                for entry in entry_graph[child]:
                    children.appendleft(entry)
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
            - ontology: a dictionary for single_pack ontology.
                default value set from env var ONTOLOGY_PATH.
            - multiOntology: a dictionary for multi_pack ontology
                default to {}.
            - projectConfigs: project configurations.
                default value generated from ontology.
        """
        config = super().default_configs()

        # TODO: Ontology path is hard coded here.
        #       Will pass ontology in a more efficient way in future version.
        with open(os.environ["ONTOLOGY_PATH"], 'r') as f:
            ontology = json.load(f)

        config.update({
            "frontend_build_path": os.environ["FRONTEND_BUILD_PATH"],
            "django_backend_path": os.environ["DJANGO_BACKEND_PATH"],
            "port": 8888,
            "host": "localhost",
            "user_name": "admin",
            "user_password": "admin",
            "projectType": "single_pack",
            "projectName": "Auto generated project",
            "ontology": ontology,
            "multiOntology": {},
            "projectConfigs": cls._default_project_configs(ontology)
        })

        return config
