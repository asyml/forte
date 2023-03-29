# Copyright 2021 The Forte Authors. All Rights Reserved.
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
Example code for pytorch blog post, a "one-liner" version.
"""
from forte.data.data_pack import DataPack
from forte.data.readers import HTMLReader
from forte.pipeline import Pipeline
from forte.processors.stave import StaveProcessor
from fortex.spacy import SpacyProcessor

Pipeline[DataPack]().set_reader(HTMLReader()).add(
    SpacyProcessor(),
    config={
        "processors": ["sentence", "tokenize", "pos", "ner", "dep", "umls_link"]
    },
).add(StaveProcessor()).run(
    "<body><p>"
    "she does not have SVS syndrome from an axillary vein thrombosis."
    "</p></body>"
)
