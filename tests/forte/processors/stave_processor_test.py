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
Unit tests for stave processor.
"""

import os
import sys
import json
import unittest
import threading
import requests

from typing import Any, Dict, Iterator, Optional, Type, Set, List
from forte.common import ProcessorConfigError
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import OntonotesReader
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor
from forte.processors.base.batch_processor import Predictor
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink

# Currently hard coded. Deprecated in future update.
os.environ["FRONTEND_BUILD_PATH"] = os.path.abspath("stave/build/")
sys.path.insert(0, os.path.abspath("stave/simple-backend/"))
from forte.processors.stave import StaveProcessor
from nlpviewer_backend.lib.stave_project import StaveProjectReader


class TestStaveProcessor(unittest.TestCase):

    def setUp(self):

        self._port: int = 8880
        self._file_dir_path = os.path.dirname(__file__)
        self._project_name: str = "serialization_pipeline_test"
        self._dataset_dir: str = os.path.abspath(os.path.join(
            self._file_dir_path, '../../../', 'data_samples/ontonotes/00/'))
        self._stave_processor = StaveProcessor()

        self.pl = Pipeline[DataPack](
            ontology_file=os.path.abspath(os.path.join(
                self._file_dir_path, "../../../",
                    "forte/ontology_specs/base_ontology.json"))
        )
        self.pl.set_reader(OntonotesReader())

    def test_stave_viewer(self):
        """
        Test in viewer mode Stave. Project data, including metadata and
        textpacks, will be dumped to local disk. Consistency checking
        is performed here to verify the dumped project data.
        """
        self.pl.add(self._stave_processor, config={
            "project_name": self._project_name,
            "server_thread_daemon": True
        })
        self.pl.run(self._dataset_dir)

        project_reader = StaveProjectReader(
            project_path=self._stave_processor.configs.project_path
        )

        self.assertEqual(project_reader.project_name, self._project_name)
        self.assertEqual(
            project_reader.project_type,
            self._stave_processor.configs.project_type
        )
        self.assertEqual(
            project_reader.ontology,
            self._stave_processor.resources.get("onto_specs_dict")
        )
        self.assertEqual(
            project_reader.project_configs,
            self._stave_processor.configs.project_configs.todict()
        )

        # Check default project configuration
        with open(os.path.abspath(os.path.join(
            self._file_dir_path, "../data/ontology/test_specs/",
            "test_project_configuration.json")), "r") as f:
            target_configs = json.load(f)
        self.assertEqual(
            target_configs,
            project_reader.project_configs,
        )

        # Check the number of newly created documents
        count, index = 0, 0
        while True:
            next_index = project_reader.get_next_index(index)
            if next_index == index:
                break
            count += 1
            index = next_index

        self.assertEqual(count + 1, len(os.listdir(self._dataset_dir)))

    def test_stave_standard(self):
        """
        Test in standard Stave. Project data will be uploaded to Stave
        backend via diango APIs. Consistency checking is performed here
        to verify the project and documents being saved to the Stave database.
        """
        self.pl.add(self._stave_processor, config={
            "port": self._port,
            "project_name": self._project_name,
            "in_viewer_mode": False,
            "server_thread_daemon": True
        })
        self.pl.run(self._dataset_dir)
        url = f"http://localhost:{self._port}"

        with requests.Session() as session:
            # Log in as admin user
            response = session.post(f"{url}/api/login",
                json={
                    "name": "admin",
                    "password": "admin"
                })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "OK")

            # Check if new project is created
            response = session.post(f"{url}/api/projects")
            self.assertEqual(response.status_code, 200)
            project_list = response.json()
            self.assertIsInstance(project_list, list)

            project_id = -1
            for project in project_list:
                if project["name"] == self._project_name:
                    project_id = project["id"]
            self.assertGreater(project_id, 0)

            # Check default project configuration
            with open(os.path.abspath(os.path.join(
                self._file_dir_path, "../data/ontology/test_specs/",
                "test_project_configuration.json")), "r") as f:
                target_configs = json.load(f)
            self.assertEqual(
                json.dumps(target_configs, sort_keys=True),
                json.dumps(
                    self._stave_processor.configs.project_configs.todict(),
                    sort_keys=True
                )
            )

            # Check the number of newly created documents
            response = session.post(f"{url}/api/projects/{project_id}/docs")
            self.assertEqual(response.status_code, 200)
            doc_list = response.json()
            self.assertIsInstance(doc_list, list)
            self.assertEqual(
                len(os.listdir(self._dataset_dir)),
                len(doc_list)
            )

    def test_projecttype_exception(self):
        """
        Check the validation of `project_type` config.
        """
        self.pl.add(self._stave_processor, config={
            "port": self._port,
            "project_type": "multi_pack",
            "server_thread_daemon": True,
            "in_viewer_mode": False
        })
        with self.assertRaises(ProcessorConfigError) as context:
            self.pl.run(self._dataset_dir)

    def test_resources_exception(self):
        """
        Check exception raised when ontology is not correctly
        configured in pipeline.
        """
        with self.assertRaises(ProcessorConfigError) as context:
            self.pl.resource.remove("onto_specs_path")
            self.pl.resource.remove("onto_specs_dict")
            self.pl.add(self._stave_processor, config={
                "port": self._port,
                "server_thread_daemon": True,
                "in_viewer_mode": False
            })
            self.pl.run(self._dataset_dir)


if __name__ == "__main__":
    unittest.main()
