#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import unittest

from ft.onto.base_ontology import Sentence, Token, EntityMention
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.extractors.seqtagging_extractor import BioSeqTaggingExtractor


class SeqTaggingExtractorTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
        # Define and config the Pipeline
        self.dataset_path = os.path.join(root_path, "data_samples/conll03")

    def test_BioSeqTaggingExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "entry_type": "ft.onto.base_ontology.EntityMention",
            "need_pad": True,
            "attribute": "ner_type",
            "tagging_unit": "ft.onto.base_ontology.Token",
        }

        expected = [
            (None, "O"),
            ("ORG", "B"),
            ("ORG", "I"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
        ]

        invalid = [
            (None, "O"),
            ("MISC", "B"),
            ("ORG", "I"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "I"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
        ]

        corrected = [
            (None, "O"),
            ("MISC", "B"),
            ("ORG", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            ("MISC", "B"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
            (None, "O"),
        ]

        extractor = BioSeqTaggingExtractor()
        extractor.initialize(config=config)

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        extractor.predefined_vocab({"MISC", "ORG"})
        invalid = [extractor.element2repr(ele) for ele in invalid]

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                feature = extractor.extract(pack, instance)
                recovered = [extractor.id2element(idx) for idx in feature._data]
                self.assertListEqual(expected, recovered)
                extractor.pre_evaluation_action(pack, instance)
                extractor.add_to_pack(pack, feature._data, instance)
            pack.add_all_remaining_entries()

            for instance in pack.get(Sentence):
                extractor.pre_evaluation_action(pack, instance)
                extractor.add_to_pack(pack, invalid, instance)
            pack.add_all_remaining_entries()

            for instance in pack.get(Sentence):
                feature = extractor.extract(pack, instance)
                recovered = [extractor.id2element(idx) for idx in feature._data]
                self.assertListEqual(corrected, recovered)


if __name__ == "__main__":
    unittest.main()
