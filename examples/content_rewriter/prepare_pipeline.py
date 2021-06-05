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
The main running pipeline for the rewriter.
"""
from reader import TableReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

from forte.processors.base import PackProcessor
from forte.processors.writers import PackNameJsonPackWriter
from ft.onto.base_ontology import Utterance


class Instructor(PackProcessor):
    def __init__(self, instruction: str):
        super().__init__()
        self.instruction = instruction

    def _process(self, input_pack: DataPack):
        input_pack.set_text(input_pack.text + '\n' + self.instruction)
        u = Utterance(input_pack,
                      len(input_pack.text) - len(self.instruction),
                      len(input_pack.text))
        u.speaker = 'ai'


instruct_text = 'This is an example to use the chatbot interface with the ' \
                'content rewriter model. To run this example, follow the ' \
                'instructions here "https://github.com/asyml/forte' \
                '/tree/master/examples/content_rewriter" to obtain ' \
                'the models and make sure Forte is in your Python Path.'

pipeline = Pipeline[DataPack]()
pipeline.set_reader(TableReader())
pipeline.add(Instructor(instruct_text))
pipeline.add(PackNameJsonPackWriter(),
             {'indent': 2, 'output_dir': 'table_inputs', 'overwrite': True,
              'drop_record': True})

pipeline.run('table_samples.txt')
