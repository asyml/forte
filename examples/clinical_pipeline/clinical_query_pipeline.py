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

from forte.data.readers import RawDataDeserializeReader
from forte.pipeline import Pipeline

from utterance_searcher import LastUtteranceSearcher

# Demonstrate how to retrieve result from the index.

with open('sample_utterance.json') as json_file:
    nlp: Pipeline = Pipeline()
    nlp.set_reader(reader=RawDataDeserializeReader())
    nlp.add(LastUtteranceSearcher())

    # Remember to initialize.
    nlp.initialize()

    # Pass data into pipeline
    data = json_file.read()
    nlp.process([data])
