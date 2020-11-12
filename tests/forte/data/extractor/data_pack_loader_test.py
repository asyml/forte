#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
import os
import shutil
import unittest

from forte.data.readers.base_reader import PackReader

from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.data_pack_loader import DataPackLoader


class DataPackLoaderTest(unittest.TestCase):
    def setUp(self):
        self.reader: PackReader = CoNLL03Reader()
        self.cache_dir: str = ".cache_test/"
        self.src_dir = "data_samples/train_pipeline_test/"
        self.file = "conll03.conll"

    def test_write_data_pack(self):
        self.data_pack_loader: DataPackLoader = \
            DataPackLoader(reader=self.reader,
                           cache_dir=self.cache_dir,
                           config={
                               "src_dir": self.src_dir,
                               "cache_writer": {
                                   "output_dir": self.cache_dir
                               }
                           })
        self.assertEqual(self.data_pack_loader._config.cache_writer.output_dir,
                         self.cache_dir)

        data_pack = next(self.reader.iter(self.src_dir))

        self.data_pack_loader._write_data_pack(data_pack)
        cache_file_path = self.cache_dir + self.src_dir + self.file + ".json"
        self.assertTrue(os.path.exists(cache_file_path))

        shutil.rmtree(self.cache_dir)

    def test_finish_1(self):
        self._execute_write_helper({
            "src_dir": self.src_dir,
            "cache_writer": {
                "output_dir": self.cache_dir
            },
            "clear_cache_after_finish": False
        })
        self.data_pack_loader.finish()
        self.assertTrue(os.path.exists(self.cache_dir))

        shutil.rmtree(self.cache_dir)

    def test_finish_2(self):
        self._execute_write_helper({
            "src_dir": self.src_dir,
            "cache_writer": {
                "output_dir": self.cache_dir
            },
            "clear_cache_after_finish": True
        })
        self.data_pack_loader.finish()
        self.assertFalse(os.path.exists(self.cache_dir))

        # Delete for safe
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _execute_write_helper(self, config):
        self.data_pack_loader: DataPackLoader = \
            DataPackLoader(reader=self.reader,
                           cache_dir=self.cache_dir,
                           config=config)

        data_pack = next(self.reader.iter(self.src_dir))
        self.data_pack_loader._write_data_pack(data_pack)


if __name__ == '__main__':
    unittest.main()
