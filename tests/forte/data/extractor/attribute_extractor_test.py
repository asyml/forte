#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import unittest
import pickle as pkl
from ft.onto.base_ontology import Sentence, Token
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.extractors.attribute_extractor import AttributeExtractor


class AttributeExtractorTest(unittest.TestCase):
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

    def test_AttributeExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config = {
            "need_pad": True,
            "entry_type": "ft.onto.base_ontology.Token",
            "attribute": "text",
        }

        extractor = AttributeExtractor()
        extractor.initialize(config=config)

        sentence = (
            "The European Commission said on Thursday it disagreed "
            "with German advice to consumers to shun British lamb "
            "until scientists determine whether mad cow disease "
            "can be transmitted to sheep ."
        )

        # Check update_vocab.
        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.update_vocab(pack, instance)

        # Check extract
        for pack in pipeline.process_dataset(self.dataset_path):
            features = []
            for instance in pack.get(Sentence):
                features.append(extractor.extract(pack, instance))

            for feat in features:
                recovered = [extractor.id2element(idx) for idx in feat.data[0]]
                self.assertEqual(" ".join(recovered), sentence)

        # Check add_to_pack and pre_evaluation_action.
        # Vocab_method is indexing, therefore the id of element
        # is the same as repr.
        extractor.config.attribute = "pos"
        extractor.add("TMP")
        fake_pos_ids = [
            extractor.element2repr("TMP")
            for _ in range(len(sentence.split(" ")))
        ]
        # After pre_evaluation_action, the attribute value will
        # become None. Since vocab_use_unk is true, None will be
        # mapped to <UNK>.
        unk_pos_ids = [
            extractor.element2repr(None)
            for _ in range(len(sentence.split(" ")))
        ]

        for pack in pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                extractor.add_to_pack(pack, instance, fake_pos_ids)

            for instance in pack.get(Sentence):
                feat = extractor.extract(pack, instance)
                self.assertEqual(feat.data[0], fake_pos_ids)

            for instance in pack.get(Sentence):
                extractor.pre_evaluation_action(pack, instance)
                feat = extractor.extract(pack, instance)
                self.assertEqual(feat.data[0], unk_pos_ids)

        # Check state and from_state.
        new_extractor = pkl.loads(pkl.dumps(extractor))
        self.assertEqual(
            new_extractor.config.attribute, extractor.config.attribute
        )


if __name__ == "__main__":
    unittest.main()
