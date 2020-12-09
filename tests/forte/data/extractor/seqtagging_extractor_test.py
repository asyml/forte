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
import unittest

from ft.onto.base_ontology import Sentence, Token, EntityMention
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.extractor.seqtagging_extractor import BioSeqTaggingExtractor


class SeqTaggingExtractorTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03_new"

    def test_BioSeqTaggingExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "scope": Sentence,
            "entry_type": EntityMention,
            "attribute": "ner_type",
            "based_on": Token,
        }

        expected = [('ORG', 'B'), (None, 'O'), ('MISC', 'B'),
                    (None, 'O'), (None, 'O'), (None, 'O'),
                    ('MISC', 'B'), (None, 'O'), (None, 'O')]

        extractor = BioSeqTaggingExtractor(config)

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                feature = extractor.extract(pack, instance)
                recovered = [extractor.id2element(idx) for idx in feature._data]
                self.assertListEqual(expected, recovered)
                extractor.add_to_pack(pack, instance, feature._data)
            pack.add_all_remaining_entries()


if __name__ == '__main__':
    unittest.main()
