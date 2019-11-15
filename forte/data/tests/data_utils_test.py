"""
Unit tests for the download utility of Forte
"""
import logging
from pathlib import Path
import os
import shutil
import unittest

from forte.data import data_utils

logging.basicConfig(level=logging.DEBUG)


class DataUtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_path = "test_dir"
        self.file_name = "test.txt"
        self.text = ["This file is used to test download utilities in Forte.\n"]

    def tearDown(self):
        shutil.rmtree(self.test_path)

    def test_data_utils(self):
        urls = ["https://drive.google.com/file/d/1YHXMiIne5MjSBePsPHPWO6hdRj4w"
                "EnSk/view?usp=sharing"]
        data_utils.maybe_download(urls=urls, path=self.test_path,
                                  filenames=[self.file_name])
        path = Path(self.test_path)
        self.assertEqual(path.exists(), True)

        files = list(os.walk(path))
        self.assertEqual(len(files), 1)

        with open(f"{self.test_path}/{self.file_name}", "r") as f:
            lines = f.readlines()
            self.assertEqual(lines, self.text)


if __name__ == '__main__':
    unittest.main()
