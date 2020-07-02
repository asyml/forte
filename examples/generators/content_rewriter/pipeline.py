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
from examples.generators.content_rewriter.rewriter import ContentRewriter
from forte.data.readers import RawDataDeserializeReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
import json
# Let's create a pipeline first.
pipeline = Pipeline[DataPack]()
pipeline.set_reader(RawDataDeserializeReader())
pipeline.add(ContentRewriter())

# You should initialize the model here, so we only do it once.
pipeline.initialize()


def do_process(input_pack_str: str):
    # Let's assume there is a JSON string for us to use.
    datapack: DataPack = pipeline.process([input_pack_str])
    # You can get the JSON form like this.
    data_json = datapack.serialize()
    # Let's print it out.
    print(data_json)
    with open('generation.txt', 'w') as f:
        f.write(data_json)

if __name__ == '__main__':
    with open('rewriting_input.json') as f:
        test_str = f.read()
        do_process(test_str)
