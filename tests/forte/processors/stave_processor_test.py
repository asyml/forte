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
"""
Unit tests for stave processor.
"""

import os
import unittest
import threading
import requests

from typing import Any, Dict, Iterator, Optional, Type, Set, List
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import OntonotesReader
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor
from forte.processors.base.batch_processor import Predictor
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink
from forte.processors import StaveProcessor
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter

forte_path = "/Users/suqi.sun/OneDrive - Petuum, Inc/Projects/forte/"
stave_path = "/Users/suqi.sun/Documents/my_stave/"


class TestStaveProcessor(unittest.TestCase):

    def setUp(self):

        os.environ["FRONTEND_BUILD_PATH"] = "stave/build/"
        os.environ["ONTOLOGY_PATH"] = "forte/ontology_specs/base_ontology.json"
        os.environ["DJANGO_BACKEND_PATH"] = "stave/simple-backend/"

        self._port: int = 8880
        self._project_name: str = "serialization_pipeline_test"

        self.pl = Pipeline[DataPack]()
        self.pl.set_reader(OntonotesReader())
        self.pl.add(NLTKSentenceSegmenter())
        self.pl.add(NLTKWordTokenizer())
        self.pl.add(NLTKPOSTagger())
        self.pl.add(StaveProcessor(), config = {
            "port": self._port,
            "projectName": self._project_name,
            "server_thread_daemon": True
        })

    def test_stave(self):

        dataset_dir = "data_samples/ontonotes/00/"

        self.pl.run(dataset_dir)
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
            
            # Check the number of newly created documents
            response = session.post(f"{url}/api/projects/{project_id}/docs")
            self.assertEqual(response.status_code, 200)
            doc_list = response.json()
            self.assertIsInstance(doc_list, list)
            self.assertEqual(len(os.listdir(dataset_dir)), len(doc_list))


if __name__ == "__main__":
    unittest.main()
