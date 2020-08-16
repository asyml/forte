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
import sys

from examples.Cliner.Cliner import ClinicalNER
from examples.Cliner.Cliner_train import CliTrain
from forte.data.data_pack import DataPack
from forte.data.readers import RawDataDeserializeReader
from forte.pipeline import Pipeline
from forte.processors.writers import PackNameJsonPackWriter

# Let's create a pipeline first.
pipeline = Pipeline[DataPack]()


def do_process(input_pack_str: str):
    pipeline.process([input_pack_str])
    # data_json = datapack.serialize()
    # with open('generation.json', 'w') as fo:
    #     fo.write(data_json)


if __name__ == '__main__':
    if sys.argv[1] == 'train':       # train mode
        model = CliTrain()
        model.train()
    else:                            # inference mode
        pipeline.set_reader(RawDataDeserializeReader())
        pipeline.add(ClinicalNER(), config={
            'config_output': sys.argv[3],
            'config_data': sys.argv[4]
        })
        pipeline.add(
            PackNameJsonPackWriter(),
            {
                'output_dir': 'output',
                'indent': 2,
                'overwrite': True,
            }
        )

        pipeline.initialize()

        with open('Cliner_input.json') as fi:
            test_str = fi.read()
            do_process(test_str)
