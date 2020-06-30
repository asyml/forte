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
from examples.generators.content_rewriter.reader import TableReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

# Let's create a pipeline that accept a Json string.
from forte.processors.writers import PackNameJsonPackWriter

pipeline = Pipeline[DataPack]()
pipeline.set_reader(TableReader())
pipeline.add(PackNameJsonPackWriter(),
             {'indent': 2, 'output_dir': '.', 'overwrite': True})
pipeline.run('| this | is | a | table', 'You can ask me questions.')
