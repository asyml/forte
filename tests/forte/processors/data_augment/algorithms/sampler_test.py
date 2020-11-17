# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Unit tests for distribution sampler.
"""
import unittest
import os
import tempfile

from ddt import ddt, data, unpack

from forte.processors.data_augment.algorithms.sampler import \
    UniformSampler, UnigramSampler


@ddt
class TestSampler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    @data((["apple 1",
            "banana 5",
            "orange 2",
            "lemon 10"],))
    @unpack
    def test_unigram_sampler(self, texts):
        file_path = os.path.join(self.test_dir, "unigram.txt")
        with open(file_path, 'w') as f:
            for text in texts:
                f.write(text + '\n')

        config = {"unigram_path": file_path}
        sampler = UnigramSampler(config)
        word = sampler.sample()
        print("word1", word)
        unigram = ('apple', 'banana', 'lemon', 'orange')
        self.assertIn(word, unigram)

    def test_uniform_sampler(self):
        config = {"distribution": "nltk"}
        sampler = UniformSampler(config)
        word = sampler.sample()
        print("word2", word)
        self.assertIn(word, sampler.vocab)


if __name__ == "__main__":
    unittest.main()
