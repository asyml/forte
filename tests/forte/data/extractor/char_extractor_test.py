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
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack import DataPack
from forte.data.extractors.char_extractor import CharExtractor


class CharExtractorTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03"

    def test_CharExtractor(self):
        pipeline = Pipeline[DataPack]()
        reader = CoNLL03Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        config1 = {
            "entry_type": Token,
            "need_pad": True,
            "vocab_use_unk": True,
        }

        config2 = {
            "entry_type": Token,
            "need_pad": True,
            "vocab_use_unk": True,
            "max_char_length": 4
        }

        for config in [config1, config2]:
            extractor = CharExtractor(config)

            sentence = "The European Commission said on Thursday it disagreed "\
                        "with German advice to consumers to shun British lamb "\
                        "until scientists determine whether mad cow disease "\
                        "can be transmitted to sheep ."

            for pack in pipeline.process_dataset(self.dataset_path):
                for instance in pack.get(Sentence):
                    extractor.update_vocab(pack, instance)

            features = []
            for pack in pipeline.process_dataset(self.dataset_path):
                for instance in pack.get(Sentence):
                    features.append(extractor.extract(pack, instance))

            for feat in features:
                recovered = [[extractor.id2element(idx) for idx in sent]
                                                for sent in feat.data[0]]

                recovered = ["".join(chars) for chars in recovered]
                recovered = " ".join(recovered)
                if "max_char_length" not in config:
                    self.assertEqual(recovered, sentence)
                else:
                    truncated_sent = [token[:config["max_char_length"]]
                            for token in sentence.split(" ")]
                    truncated_sent = " ".join(truncated_sent)
                    self.assertEqual(recovered, truncated_sent)


if __name__ == '__main__':
    unittest.main()
