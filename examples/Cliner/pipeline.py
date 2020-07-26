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
import json

from examples.Cliner.Cliner import ClinicalNER
from examples.Cliner.Cliner_train import CliTrain
from forte.data.data_pack import DataPack
from forte.data.readers import RawDataDeserializeReader
from forte.pipeline import Pipeline

# Let's create a pipeline first.
pipeline = Pipeline[DataPack]()


def do_process(input_pack_str: str):
    # Let's assume there is a JSON string for us to use.
    datapack: DataPack = pipeline.process([input_pack_str])
    # You can get the JSON form like this.
    data_json = datapack.serialize()
    # Let's write it out.
    with open('generation.json', 'w') as fo:
        fo.write(json.dumps(data_json, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'train':       # train mode
        model = CliTrain()
        model.train()
    else:                            # inference mode
        pipeline.set_reader(RawDataDeserializeReader())
        pipeline.add(ClinicalNER())

        # You should initialize the model here, so we only do it once.
        pipeline.initialize()

        with open('Cliner_input.json') as fi:
            test_str = fi.read()
            do_process(test_str)
