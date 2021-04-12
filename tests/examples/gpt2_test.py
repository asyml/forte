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
Unit tests for GPT2 Example.
"""
import os
import unittest

from examples.gpt2 import multipack_pipeline_gpt2


class Gpt2Test(unittest.TestCase):

    def test_pipeline_construction(self):
        module_path = multipack_pipeline_gpt2.__file__
        sample_config = os.path.join(os.path.dirname(module_path),
                                     'sample_multipack_pipeline_gpt.yml')
        p = multipack_pipeline_gpt2.create_pipeline(sample_config)

        self.assertEqual(len(p.components), 3)

        # pylint: disable=line-too-long
        p_names = [
            'forte.processors.third_party.text_generation_processor.TextGenerationProcessor',
            'forte.processors.third_party.nltk_processors.NLTKWordTokenizer',
            'forte.processors.third_party.nltk_processors.NLTKPOSTagger',
        ]

        for proc, p_name in zip(p.components, p_names):
            self.assertEqual(proc.name, p_name)

        self.assertEqual(
            p.reader.name,
            'forte.data.readers.multipack_sentence_reader.'
            'MultiPackSentenceReader')


if __name__ == '__main__':
    unittest.main()
