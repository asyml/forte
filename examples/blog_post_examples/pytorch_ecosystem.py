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
Example code for pytorch blog post
"""

from forte.data.readers import HTMLReader
from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.processors.stave import StaveProcessor
from fortex.spacy import SpacyProcessor

from ft.onto.base_ontology import Dependency
from ftx.medical.clinical import MedicalEntityMention


def main():

    # Create input HTML string
    input_html = (
        "<body><p>"
        "she does not have SVS syndrome from an axillary vein thrombosis."
        "</p></body>"
    )


    # Assemble the pipeline:
    # A pipeline is consisted of a set of Components (readers and processors).
    # The data flows in the pipeline as data packs, and each component will
    # use or add information to the data packs.
    pipeline = Pipeline[DataPack]()

    # Set HTMLReader as pipeline reader:
    # HTMLReader takes in list of html strings, cleans the HTML tags and
    # stores the cleaned text in pack.
    pipeline.set_reader(HTMLReader())

    # Add SpacyProcessor to process the datapack:
    # SpacyProcessor provides functions including sentence parsing, tokenize,
    # POS tagging, lemmatization, NER, and medical entity linking. This
    # processor will do user defined tasks according to configs.
    pipeline.add(SpacyProcessor(), config={
        "processors": ["sentence", "tokenize", "pos", "ner", "dep", "umls_link"]
    })

    # Add StaveProcessor to visualize the processing results：
    # StaveProcessor provides easy visualization for forte users. We can
    # visualize datapack with annotations by inserting it into the forte
    # pipeline.
    pipeline.add(StaveProcessor())

    # Initialize and run the pipeline
    pipeline.initialize()
    for datapack in pipeline.process_dataset(input_html):

        # Get the results of dependency parsing
        for link in datapack.get(Dependency):
            parent = link.get_parent()
            child = link.get_child()
            # Print out the dependency between parent and child
            print(
                f"'{child.text}' has dependency '{link.dep_label}'"
                f" of parent '{parent.text}'"
            )

        # Retrieve all the MedicalEntityMention in datapack
        for medical_entity in datapack.get(MedicalEntityMention):
            # Get and print out the UMLS links
            for umls_link in medical_entity.umls_entities:
                print(umls_link)


if __name__ == "__main__":
    main()
