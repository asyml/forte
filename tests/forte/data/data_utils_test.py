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
Unit tests for the download utility of Forte
"""
from pathlib import Path

import os
import shutil
import unittest
from requests import HTTPError
import logging

from forte.data import data_utils


logger = logging.getLogger(__name__)


class DataUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_path = "test_dir"
        self.file_name = "test.txt"
        self.text = ["This file is used to test download utilities in Forte.\n"]

    def tearDown(self):
        shutil.rmtree(self.test_path)

    def test_data_utils(self):
        urls = [
            "https://drive.google.com/file/d/1YHXMiIne5MjSBePsPHPWO6hdRj4w"
            "EnSk/view?usp=sharing"
        ]
        try:
            data_utils.maybe_download(
                urls=urls, path=self.test_path, filenames=[self.file_name]
            )
        except HTTPError as e:
            if e.response.status_code != 403:
                raise e
            logger.warning(
                "Skipping HTTPError from %s", e.response.url, exc_info=True
            )
            return
        path = Path(self.test_path)
        self.assertEqual(path.exists(), True)

        files = list(os.walk(path))
        self.assertEqual(len(files), 1)

        with open(f"{self.test_path}/{self.file_name}", "r") as f:
            lines = f.readlines()
            self.assertEqual(lines, self.text)


if __name__ == "__main__":
    unittest.main()
