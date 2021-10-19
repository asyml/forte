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
Example code for "Building an Information System for Clinical Notes"
"""
from fortex.spacy import SpacyProcessor
from forte.data.readers import HTMLReader
from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.processors.stave import StaveProcessor
from ft.onto.base_ontology import Dependency
from ftx.medical import MedicalEntityMention


def main():

    # Create input HTML string
    input_html = (
        "<body><p>"
        "she does not have SVS syndrome from an axillary vein thrombosis."
        "</p></body>"
    )

    # Assemble the pipeline
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(HTMLReader())
    pipeline.add(SpacyProcessor(), config={
        "processors": ["sentence", "tokenize", "pos", "ner", "dep", "umls_link"]
    })
    pipeline.add(StaveProcessor())

    # Initialize and run the pipeline
    pipeline.initialize()
    for datapack in pipeline.process_dataset(input_html):

        # Get the results of dependency parsing
        for link in datapack.get(Dependency):
            parent = link.get_parent()
            child = link.get_child()
            print(
                f"'{child.text}' has dependency '{link.dep_label}'"
                f" of parent '{parent.text}'"
            )

        # Get the UMLS links
        for medical_entity in datapack.get(MedicalEntityMention):
            for umls_link in medical_entity.umls_entities:
                print(umls_link)


if __name__ == "__main__":
    main()
